#include "filter/filter.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

extern "C" {
    // CUDA wrapper launchers defined in src/cuda/filter_kernels.cu
    void launch_count_cells_per_gene(
        long long nnz,
        const int32_t* col_idx,
        const float* data,
        float          cutoff,
        int32_t* out_counts);

    void launch_ngene_per_cell(
        long long n_rows,
        const long long* indptr,
        const int32_t* indices,
        const float* data,
        const uint8_t* keep_gene,
        int32_t* out_counts);

    void launch_max_per_gene_masked(
        long long n_rows,
        const long long* indptr,
        const int32_t* indices,
        const float* data,
        const uint8_t* keep_cell,
        const uint8_t* keep_gene,
        float* gene_max);
}

#define CUDA_CHECK(call) do {                                            \
  cudaError_t err__ = (call);                                            \
  if (err__ != cudaSuccess) {                                            \
    throw std::runtime_error(std::string("CUDA error: ") +               \
      cudaGetErrorString(err__) + " at " + __FILE__ + ":" +              \
      std::to_string(__LINE__));                                         \
  }                                                                      \
} while(0)

namespace fs = std::filesystem;

namespace rarecell {

    // ------------------------------------------------------------------
    // Ctor
    // ------------------------------------------------------------------
    FilterEngine::FilterEngine(MPI_Comm comm, const H5ADReadResult& R, const FilterConfig& cfg)
        : comm_(comm), R_(R), cfg_(cfg) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_);
        n_rows_local_ = R_.X_local.row1 - R_.X_local.row0;
        nnz_local_ = static_cast<int64_t>(R_.X_local.data.size());
        n_cols_ = R_.X_local.n_cols;
    }

    // ------------------------------------------------------------------
    // Device init: choose GPU per node-local rank, warm up, print mem/estimate
    // ------------------------------------------------------------------
    void FilterEngine::init_cuda_device() {
        int dev_count = 0;
        cudaError_t e = cudaGetDeviceCount(&dev_count);
        if (e != cudaSuccess || dev_count <= 0) {
            throw std::runtime_error(std::string("No CUDA device available: ") + cudaGetErrorString(e));
        }
        // Node-local rank for device mapping
        MPI_Comm local_comm;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local_comm);
        int local_rank = 0;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);

        device_id_ = local_rank % dev_count;
        

        CUDA_CHECK(cudaSetDevice(device_id_));
        // Create context early to surface errors
        CUDA_CHECK(cudaFree(0));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

        // Estimate upcoming allocations
        const uint64_t bytes_indptr = (uint64_t)(n_rows_local_ + 1) * sizeof(int64_t);
        const uint64_t bytes_idx = (uint64_t)nnz_local_ * sizeof(int32_t);
        const uint64_t bytes_data = (uint64_t)nnz_local_ * sizeof(float);
        const uint64_t bytes_counts = (uint64_t)n_cols_ * sizeof(int32_t);
        const uint64_t bytes_keepg = (uint64_t)n_cols_ * sizeof(uint8_t);
        const uint64_t bytes_ngenes = (uint64_t)n_rows_local_ * sizeof(int32_t);
        const uint64_t bytes_keepc = (uint64_t)n_rows_local_ * sizeof(uint8_t);
        const uint64_t bytes_gmax = (uint64_t)n_cols_ * sizeof(float);
        const uint64_t bytes_total_est =
            bytes_indptr + bytes_idx + bytes_data +
            bytes_counts + bytes_keepg + bytes_ngenes + bytes_keepc + bytes_gmax;

        std::cout << "[rank " << rank_ << "] GPU " << device_id_ << " (" << prop.name
            << ") CC " << prop.major << "." << prop.minor
            << " | free " << (free_b / (1024.0 * 1024.0)) << " MiB"
            << " / total " << (total_b / (1024.0 * 1024.0)) << " MiB"
            << " | est alloc ~ " << (bytes_total_est / (1024.0 * 1024.0)) << " MiB"
            << std::endl;

        // Safety check (leave ~10% headroom)
        const double safety = 0.10;
        if (bytes_total_est > (uint64_t)((1.0 - safety) * (double)free_b)) {
            throw std::runtime_error(
                "Estimated GPU memory usage exceeds available VRAM on device " +
                std::to_string(device_id_) + ". Reduce ranks per GPU or shard further.");
        }
    }

    // ------------------------------------------------------------------
    // Device alloc / free
    // ------------------------------------------------------------------
    static inline void safe_cuda_memcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
        if (n == 0) return;
        CUDA_CHECK(cudaMemcpy(dst, src, n, kind));
    }

    static void dump_local_csr_state(int rank,
        int64_t n_rows_local,
        int64_t nnz_local,
        int64_t n_cols,
        const CSRMatrixF32& X) {
        auto indptr_sz = (int64_t)X.indptr.size();
        auto indices_sz = (int64_t)X.indices.size();
        auto data_sz = (int64_t)X.data.size();
        std::cout << "[rank " << rank << "] shard: rows_local=" << n_rows_local
            << " nnz_local=" << nnz_local
            << " n_cols=" << n_cols
            << " | sizes: indptr=" << indptr_sz
            << " indices=" << indices_sz
            << " data=" << data_sz << "\n";
    }

    void FilterEngine::device_alloc() {
        int cur = -1;
        CUDA_CHECK(cudaGetDevice(&cur));
        if (cur != device_id_) throw std::runtime_error("cudaSetDevice did not stick");

        // --------- PRE-FLIGHT: check CSR shape consistency ---------
        dump_local_csr_state(rank_, n_rows_local_, nnz_local_, n_cols_, R_.X_local);

        if ((int64_t)R_.X_local.indptr.size() != (n_rows_local_ + 1)) {
            throw std::runtime_error("CSR indptr size mismatch: expected n_rows_local+1="
                + std::to_string(n_rows_local_ + 1) + " got " + std::to_string(R_.X_local.indptr.size()));
        }
        if ((int64_t)R_.X_local.indices.size() != nnz_local_) {
            throw std::runtime_error("CSR indices size mismatch: expected nnz_local="
                + std::to_string(nnz_local_) + " got " + std::to_string(R_.X_local.indices.size()));
        }
        if ((int64_t)R_.X_local.data.size() != nnz_local_) {
            throw std::runtime_error("CSR data size mismatch: expected nnz_local="
                + std::to_string(nnz_local_) + " got " + std::to_string(R_.X_local.data.size()));
        }

        // Short-circuit if there is literally no work
        if (n_rows_local_ == 0 && nnz_local_ == 0 && n_cols_ == 0) return;

        // --------- Allocate device buffers ---------
        CUDA_CHECK(cudaMalloc(&d_indptr_, (n_rows_local_ + 1) * sizeof(int64_t)));
        if (nnz_local_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_indices_, nnz_local_ * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_data_, nnz_local_ * sizeof(float)));
        }
        else {
            d_indices_ = nullptr; d_data_ = nullptr;
        }

        CUDA_CHECK(cudaMalloc(&d_counts_gene_, std::max<int64_t>(n_cols_, 1) * (int64_t)sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_counts_gene_, 0, n_cols_ * sizeof(int32_t)));

        CUDA_CHECK(cudaMalloc(&d_keep_gene_, std::max<int64_t>(n_cols_, 1) * (int64_t)sizeof(uint8_t)));
        if (n_cols_ > 0) CUDA_CHECK(cudaMemset(d_keep_gene_, 0, n_cols_ * sizeof(uint8_t)));

        CUDA_CHECK(cudaMalloc(&d_ngenes_cell_, std::max<int64_t>(n_rows_local_, 1) * (int64_t)sizeof(int32_t)));

        CUDA_CHECK(cudaMalloc(&d_keep_cell_, std::max<int64_t>(n_rows_local_, 1) * (int64_t)sizeof(uint8_t)));
        if (n_rows_local_ > 0) CUDA_CHECK(cudaMemset(d_keep_cell_, 0, n_rows_local_ * sizeof(uint8_t)));

        CUDA_CHECK(cudaMalloc(&d_gene_max_, std::max<int64_t>(n_cols_, 1) * (int64_t)sizeof(float)));
        if (n_cols_ > 0) CUDA_CHECK(cudaMemset(d_gene_max_, 0, n_cols_ * sizeof(float)));

        // --------- Labeled, chunked copies (so we know exactly where it fails) ---------
        auto copy_chunked = [&](const char* label, void* dst, const void* src, size_t bytes) {
            if (bytes == 0) return;
            if (!dst) throw std::runtime_error(std::string("cudaMemcpy: dst is null for ") + label);
            if (!src) throw std::runtime_error(std::string("cudaMemcpy: src is null for ") + label);

            // Print once per buffer
            std::cout << "[rank " << rank_ << "] H2D " << label
                << " bytes=" << (double)bytes / (1024.0 * 1024.0) << " MiB\n";

            const size_t CHUNK = size_t(256) * 1024 * 1024; // 256 MiB
            size_t off = 0;
            while (off < bytes) {
                size_t n = std::min(CHUNK, bytes - off);
                CUDA_CHECK(cudaMemcpy(static_cast<char*>(dst) + off,
                    static_cast<const char*>(src) + off,
                    n, cudaMemcpyHostToDevice));
                off += n;
            }
            };

        // Copies
        copy_chunked("indptr", d_indptr_,
            R_.X_local.indptr.data(), (size_t)(n_rows_local_ + 1) * sizeof(int64_t));

        if (nnz_local_ > 0) {
            copy_chunked("indices", d_indices_,
                R_.X_local.indices.data(), (size_t)nnz_local_ * sizeof(int32_t));
            copy_chunked("data", d_data_,
                R_.X_local.data.data(), (size_t)nnz_local_ * sizeof(float));
        }
    }

    void FilterEngine::device_free() {
        if (d_indptr_)      cudaFree(d_indptr_);
        if (d_indices_)     cudaFree(d_indices_);
        if (d_data_)        cudaFree(d_data_);
        if (d_counts_gene_) cudaFree(d_counts_gene_);
        if (d_keep_gene_)   cudaFree(d_keep_gene_);
        if (d_ngenes_cell_) cudaFree(d_ngenes_cell_);
        if (d_keep_cell_)   cudaFree(d_keep_cell_);
        if (d_gene_max_)    cudaFree(d_gene_max_);
        d_indptr_ = nullptr; d_indices_ = nullptr; d_data_ = nullptr;
        d_counts_gene_ = nullptr; d_keep_gene_ = nullptr; d_ngenes_cell_ = nullptr;
        d_keep_cell_ = nullptr; d_gene_max_ = nullptr;
    }

    // ------------------------------------------------------------------
    // Step 1: per-gene counts of cells with value > expression_cutoff (GPU + Allreduce)
    // ------------------------------------------------------------------
    void FilterEngine::step_count_cells_per_gene() {
        launch_count_cells_per_gene(nnz_local_, d_indices_, d_data_,
            cfg_.expression_cutoff, d_counts_gene_);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local counts and reduce across ranks
        std::vector<int32_t> local_counts((size_t)n_cols_, 0);
        if (n_cols_ > 0) {
            CUDA_CHECK(cudaMemcpy(local_counts.data(), d_counts_gene_, n_cols_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        ncells_per_gene_global_.assign((size_t)n_cols_, 0);
        if (n_cols_ > 0) {
            MPI_Allreduce(local_counts.data(), ncells_per_gene_global_.data(),
                (int)n_cols_, MPI_INT, MPI_SUM, comm_);
        }
    }

    // ------------------------------------------------------------------
    // Step 2: initial keep_gene = !(MIR|Mir) AND (ncells_per_gene >= min_cells)
    // ------------------------------------------------------------------
    static inline bool contains_MIR_or_Mir(const std::string& s) {
        // mirrors Python regex: "MIR|Mir"
        return s.find("MIR") != std::string::npos || s.find("Mir") != std::string::npos;
    }

    void FilterEngine::step_build_initial_keep_gene() {
        keep_gene_mask_.assign((size_t)n_cols_, 0);

        for (int64_t g = 0; g < n_cols_; ++g) {
            bool non_mir = true;
            if (cfg_.remove_mir) {
                non_mir = !contains_MIR_or_Mir(R_.gene_names[(size_t)g]);
            }
            bool enough_cells = (ncells_per_gene_global_[(size_t)g] >= cfg_.min_cells);
            keep_gene_mask_[(size_t)g] = (non_mir && enough_cells) ? 1 : 0;
        }

        if (n_cols_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_keep_gene_, keep_gene_mask_.data(),
                n_cols_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
    }

    // ------------------------------------------------------------------
    // Step 3: per-cell counts of kept genes > 0; keep cells with count >= min_genes
    // ------------------------------------------------------------------
    void FilterEngine::step_ngene_per_cell() {
        if (n_rows_local_ == 0) {
            keep_cell_mask_.assign(0, 0);
            return;
        }

        launch_ngene_per_cell(n_rows_local_, d_indptr_, d_indices_, d_data_,
            d_keep_gene_, d_ngenes_cell_);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int32_t> local_counts((size_t)n_rows_local_, 0);
        CUDA_CHECK(cudaMemcpy(local_counts.data(), d_ngenes_cell_,
            n_rows_local_ * sizeof(int32_t), cudaMemcpyDeviceToHost));

        keep_cell_mask_.assign((size_t)n_rows_local_, 0);
        for (int64_t r = 0; r < n_rows_local_; ++r) {
            keep_cell_mask_[(size_t)r] = (local_counts[(size_t)r] >= cfg_.min_genes) ? 1 : 0;
        }

        CUDA_CHECK(cudaMemcpy(d_keep_cell_, keep_cell_mask_.data(),
            n_rows_local_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------
    // Step 4: per-gene maxima over kept cells & kept genes; reduce MAX across ranks
    // ------------------------------------------------------------------
    void FilterEngine::step_gene_max_on_kept() {
        if (n_rows_local_ > 0) {
            launch_max_per_gene_masked(n_rows_local_, d_indptr_, d_indices_, d_data_,
                d_keep_cell_, d_keep_gene_, d_gene_max_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> gene_max_local((size_t)n_cols_, 0.0f);
        if (n_cols_ > 0) {
            CUDA_CHECK(cudaMemcpy(gene_max_local.data(), d_gene_max_,
                n_cols_ * sizeof(float), cudaMemcpyDeviceToHost));
        }

        gene_max_global_.assign((size_t)n_cols_, 0.0f);
        if (n_cols_ > 0) {
            MPI_Allreduce(gene_max_local.data(), gene_max_global_.data(),
                (int)n_cols_, MPI_FLOAT, MPI_MAX, comm_);
        }
    }

    // ------------------------------------------------------------------
    // Step 5: apply log2(max + 0.1) band to genes
    // ------------------------------------------------------------------
    void FilterEngine::step_apply_log2_band() {
        for (int64_t g = 0; g < n_cols_; ++g) {
            if (!keep_gene_mask_[(size_t)g]) continue;
            float mx = gene_max_global_[(size_t)g];
            float v = std::log2(mx + 0.1f);
            if (!(v >= cfg_.log2_cutoffl && v <= cfg_.log2_cutoffh)) {
                keep_gene_mask_[(size_t)g] = 0;
            }
        }
        if (n_cols_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_keep_gene_, keep_gene_mask_.data(),
                n_cols_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
    }

    // ------------------------------------------------------------------
    // Step 6: materialize filtered CSR (CPU compaction), optional CPM, write outputs
    // ------------------------------------------------------------------
    void FilterEngine::step_cpm_normalize(CSRMatrixF32& X) {
        // Per-row library size and scale to CPM
        const int64_t nloc = X.row1 - X.row0;
        std::vector<double> lib((size_t)nloc, 0.0);
        for (int64_t r = 0; r < nloc; ++r) {
            int64_t s = X.indptr[(size_t)r];
            int64_t e = X.indptr[(size_t)r + 1];
            double sum = 0.0;
            for (int64_t p = s; p < e; ++p) sum += (double)X.data[(size_t)p];
            lib[(size_t)r] = (sum > 0.0) ? sum : 1.0;
        }
        const double scale = 1'000'000.0;
        for (int64_t r = 0; r < nloc; ++r) {
            int64_t s = X.indptr[(size_t)r];
            int64_t e = X.indptr[(size_t)r + 1];
            double mul = scale / lib[(size_t)r];
            for (int64_t p = s; p < e; ++p) X.data[(size_t)p] = (float)((double)X.data[(size_t)p] * mul);
        }
    }

    FilterOutputs FilterEngine::step_materialize_filtered() {
        FilterOutputs out;
        out.keep_gene_final = keep_gene_mask_;
        out.keep_cell_local = keep_cell_mask_;

        // Column remap
        out.col_remap_final.assign((size_t)n_cols_, -1);
        int32_t newc = 0;
        for (int64_t g = 0; g < n_cols_; ++g) {
            if (keep_gene_mask_[(size_t)g]) out.col_remap_final[(size_t)g] = newc++;
        }

        // Kept local rows & IDs
        for (int64_t r = 0; r < n_rows_local_; ++r) {
            if (keep_cell_mask_[(size_t)r]) out.kept_local_rows.push_back(r);
        }
        out.cell_ids_local_filtered.reserve(out.kept_local_rows.size());
        for (auto r : out.kept_local_rows) {
            out.cell_ids_local_filtered.push_back(R_.cell_ids_local[(size_t)r]);
        }

        // Filtered gene names (replicated)
        out.gene_names_filtered.reserve((size_t)newc);
        for (int64_t g = 0; g < n_cols_; ++g) {
            if (keep_gene_mask_[(size_t)g]) out.gene_names_filtered.push_back(R_.gene_names[(size_t)g]);
        }

        // Compact CSR on CPU
        const auto& Xin = R_.X_local;
        CSRMatrixF32 Xout;
        Xout.indptr.resize(out.kept_local_rows.size() + 1);
        Xout.indptr[0] = 0;
        Xout.indices.clear(); Xout.data.clear();

        int64_t write_ptr = 0;
        for (size_t i = 0; i < out.kept_local_rows.size(); ++i) {
            int64_t r = out.kept_local_rows[i];
            int64_t s = Xin.indptr[(size_t)r];
            int64_t e = Xin.indptr[(size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t oldc = Xin.indices[(size_t)p];
                int32_t newcol = out.col_remap_final[(size_t)oldc];
                if (newcol >= 0) {
                    Xout.indices.push_back(newcol);
                    Xout.data.push_back(Xin.data[(size_t)p]);
                    ++write_ptr;
                }
            }
            Xout.indptr[i + 1] = write_ptr;
        }

        // Set filtered shape (local view)
        Xout.n_cols = newc;
        Xout.n_rows = (int64_t)out.kept_local_rows.size();
        Xout.row0 = 0;
        Xout.row1 = Xout.n_rows;

        out.X_local_filtered = std::move(Xout);

        // Optional CPM
        if (cfg_.preprocess_method == "cpm") {
            step_cpm_normalize(out.X_local_filtered);
        }

        // Write outputs (per-rank files)
        if (!fs::exists(cfg_.output_folder)) {
            try { fs::create_directories(cfg_.output_folder); }
            catch (...) {}
        }

        // Rank 0 writes filtered gene list
        if (rank_ == 0) {
            std::ofstream gfile(fs::path(cfg_.output_folder) / "filtered_genes.txt");
            for (auto& g : out.gene_names_filtered) gfile << g << "\n";
        }
        // Each rank writes its kept cell IDs
        {
            std::ofstream cfile(fs::path(cfg_.output_folder) / ("filtered_cells_rank" + std::to_string(rank_) + ".txt"));
            for (auto& cid : out.cell_ids_local_filtered) cfile << cid << "\n";
        }

        // COO triplets (recommended at scale)
        if (cfg_.write_coo_csv) {
            auto& X = out.X_local_filtered;
            auto path = fs::path(cfg_.output_folder) / ("filtered_counts_rank" + std::to_string(rank_) + "_coo.csv");
            std::ofstream f(path);
            f << "cell_id,gene,value\n";
            for (int64_t r = 0; r < (int64_t)out.cell_ids_local_filtered.size(); ++r) {
                const char* cid = out.cell_ids_local_filtered[(size_t)r].c_str();
                int64_t s = X.indptr[(size_t)r];
                int64_t e = X.indptr[(size_t)r + 1];
                for (int64_t p = s; p < e; ++p) {
                    int32_t g = X.indices[(size_t)p];
                    const char* gname = out.gene_names_filtered[(size_t)g].c_str();
                    float v = X.data[(size_t)p];
                    f << cid << "," << gname << "," << v << "\n";
                }
            }
        }

        // Dense CSV (debug / small data only)
        if (cfg_.write_dense_csv) {
            auto& X = out.X_local_filtered;
            auto path = fs::path(cfg_.output_folder) / ("filtered_counts_rank" + std::to_string(rank_) + ".csv");
            std::ofstream f(path);
            // Header
            f << "gene";
            for (auto& cid : out.cell_ids_local_filtered) f << "," << cid;
            f << "\n";
            // Naive sparse->dense per gene (OK for small local slices)
            const int64_t n_genes = (int64_t)out.gene_names_filtered.size();
            std::vector<float> rowbuf(out.cell_ids_local_filtered.size(), 0.0f);
            for (int64_t g = 0; g < n_genes; ++g) {
                std::fill(rowbuf.begin(), rowbuf.end(), 0.0f);
                for (int64_t r = 0; r < (int64_t)out.cell_ids_local_filtered.size(); ++r) {
                    int64_t s = X.indptr[(size_t)r];
                    int64_t e = X.indptr[(size_t)r + 1];
                    for (int64_t p = s; p < e; ++p) {
                        if (X.indices[(size_t)p] == g) rowbuf[(size_t)r] = X.data[(size_t)p];
                    }
                }
                f << out.gene_names_filtered[(size_t)g];
                for (float v : rowbuf) f << "," << v;
                f << "\n";
            }
        }

        return out;
    }

    // ------------------------------------------------------------------
    // Run orchestrator
    // ------------------------------------------------------------------
    FilterOutputs FilterEngine::run() {
        // Prepare a device for this rank and warm it up
        init_cuda_device();

        // Allocate device buffers and copy CSR
        device_alloc();

        // Steps
        step_count_cells_per_gene();
        step_build_initial_keep_gene();
        step_ngene_per_cell();
        step_gene_max_on_kept();
        step_apply_log2_band();
        auto outs = step_materialize_filtered();

        // Cleanup device
        device_free();

        return outs;
    }

} // namespace rarecell
