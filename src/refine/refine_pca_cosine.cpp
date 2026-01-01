#include "refine/refine_pca_cosine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

extern "C" {
#include <igraph/igraph.h>
#if __has_include(<igraph/igraph_version.h>)
#include <igraph/igraph_version.h>
#endif
}

namespace rarecell {

    // -------------------- small helpers --------------------

    static inline void mpi_check(int rc, const char* msg) {
        if (rc != MPI_SUCCESS) throw std::runtime_error(std::string("MPI error: ") + msg);
    }

    static inline void igraph_check(igraph_error_t rc, const char* where) {
        if (rc != IGRAPH_SUCCESS) {
            std::string s = "igraph error at ";
            s += where;
            s += ": ";
            s += igraph_strerror(rc);
            throw std::runtime_error(s);
        }
    }
    static inline void igraph_check(int rc, const char* where) {
        igraph_check((igraph_error_t)rc, where);
    }

    static std::vector<int32_t> allgather_labels(MPI_Comm comm,
        const std::vector<int32_t>& labels_local,
        int64_t local_rows,
        int64_t row0_global,
        int64_t N_global)
    {
        int rank = 0, world = 1;
        mpi_check(MPI_Comm_rank(comm, &rank), "Comm_rank");
        mpi_check(MPI_Comm_size(comm, &world), "Comm_size");

        // gather row counts
        int local_n = (int)local_rows;
        std::vector<int> counts, displs;
        counts.resize(world);
        displs.resize(world);

        mpi_check(MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm),
            "Allgather(counts)");

        displs[0] = 0;
        for (int r = 1; r < world; ++r) displs[r] = displs[r - 1] + counts[r - 1];
        const int total_n = displs[world - 1] + counts[world - 1];
        if ((int64_t)total_n != N_global) {
            // still proceed, but it's a hard mismatch
            throw std::runtime_error("allgather_labels: N_global mismatch vs gathered rows.");
        }

        std::vector<int32_t> labels_global((size_t)total_n);
        static_assert(sizeof(int) == 4, "MPI_INT assumed 32-bit");

        mpi_check(MPI_Allgatherv((void*)labels_local.data(), local_n, MPI_INT,
            (void*)labels_global.data(), counts.data(), displs.data(), MPI_INT,
            comm),
            "Allgatherv(labels)");

        (void)row0_global;
        return labels_global;
    }

    // Map gene names -> indices in gene_names. Returns selected indices (0..n_genes-1) and also a dense map gene->selpos.
    static void map_refine_genes(const std::vector<std::string>& gene_names,
        const std::vector<std::string>& refine_gene_set,
        std::vector<int32_t>& sel_gene_idx,      // global gene indices
        std::vector<int32_t>& gene_to_selpos)    // size n_genes, -1 or [0..m)
    {
        const int64_t G = (int64_t)gene_names.size();
        gene_to_selpos.assign((size_t)G, -1);

        std::unordered_map<std::string, int32_t> name2idx;
        name2idx.reserve((size_t)G * 2);
        for (int32_t i = 0; i < (int32_t)G; ++i) name2idx[gene_names[(size_t)i]] = i;

        sel_gene_idx.clear();
        sel_gene_idx.reserve(refine_gene_set.size());

        for (const auto& g : refine_gene_set) {
            auto it = name2idx.find(g);
            if (it == name2idx.end()) continue;
            sel_gene_idx.push_back(it->second);
        }

        // unique
        std::sort(sel_gene_idx.begin(), sel_gene_idx.end());
        sel_gene_idx.erase(std::unique(sel_gene_idx.begin(), sel_gene_idx.end()), sel_gene_idx.end());

        for (int32_t pos = 0; pos < (int32_t)sel_gene_idx.size(); ++pos) {
            gene_to_selpos[(size_t)sel_gene_idx[(size_t)pos]] = pos;
        }
    }

    // Pack a parent cluster’s local rows into CSR (rows = cluster cells on this rank, cols = selected genes [0..m))
    static void build_local_cluster_csr_and_candidates(
        const CSRMatrixF32& X_local,
        const std::vector<int32_t>& labels_major_local,
        const std::vector<int32_t>& gene_to_selpos,
        int32_t parent_label,
        const KNNGraphLocal* G_global,
        const std::vector<int32_t>& labels_major_global, // for neighbor membership test
        int64_t row0_global,
        // outputs:
        std::vector<int64_t>& cell_ids_local,
        std::vector<int64_t>& X_indptr_local,
        std::vector<int32_t>& X_indices_local,
        std::vector<float>& X_data_local,
        std::vector<int64_t>& Cand_indptr_local,
        std::vector<int64_t>& Cand_indices_global_local,
        std::vector<float>& Cand_w_local
    ) {
        const int64_t local_rows = (int64_t)labels_major_local.size();
        const int64_t n_genes = X_local.n_cols;

        const int32_t m = (int32_t)std::count_if(gene_to_selpos.begin(), gene_to_selpos.end(),
            [](int32_t v) { return v >= 0; });

        (void)m; // used implicitly; cols are encoded in indices_local as selpos.

        cell_ids_local.clear();
        X_indptr_local.clear();
        X_indices_local.clear();
        X_data_local.clear();

        Cand_indptr_local.clear();
        Cand_indices_global_local.clear();
        Cand_w_local.clear();

        // collect local cells in this parent
        std::vector<int64_t> local_cells;
        local_cells.reserve((size_t)local_rows);
        for (int64_t r = 0; r < local_rows; ++r) {
            if (labels_major_local[(size_t)r] == parent_label) local_cells.push_back(r);
        }

        const int64_t n_local = (int64_t)local_cells.size();
        cell_ids_local.resize((size_t)n_local);
        X_indptr_local.assign((size_t)n_local + 1, 0);
        Cand_indptr_local.assign((size_t)n_local + 1, 0);

        // build X CSR and candidates
        int64_t nnz_x = 0;
        int64_t nnz_cand = 0;

        for (int64_t i = 0; i < n_local; ++i) {
            const int64_t r = local_cells[(size_t)i];
            const int64_t gid = row0_global + r;
            cell_ids_local[(size_t)i] = gid;

            // expression slice
            const int64_t s = X_local.indptr[(size_t)r];
            const int64_t e = X_local.indptr[(size_t)r + 1];

            for (int64_t p = s; p < e; ++p) {
                const int32_t g = X_local.indices[(size_t)p];
                if (g < 0 || (int64_t)g >= n_genes) continue;
                const int32_t selpos = gene_to_selpos[(size_t)g];
                if (selpos < 0) continue;
                const float v = X_local.data[(size_t)p];
                if (v == 0.0f) continue;
                X_indices_local.push_back(selpos);
                X_data_local.push_back(v);
                ++nnz_x;
            }
            X_indptr_local[(size_t)i + 1] = nnz_x;

            // candidate edges from global graph
            if (G_global) {
                // local row of G_global corresponds to this rank’s rows; assume same partitioning by row0/row1
                // We require G_global->row0 == row0_global and row1 == row0_global+local_rows for correctness in this use.
                const int64_t gr = r; // local row index
                const int64_t gs2 = G_global->indptr[(size_t)gr];
                const int64_t ge2 = G_global->indptr[(size_t)gr + 1];
                for (int64_t q = gs2; q < ge2; ++q) {
                    const int64_t j = G_global->indices[(size_t)q]; // global id
                    if (j < 0 || j >= (int64_t)labels_major_global.size()) continue;
                    if (labels_major_global[(size_t)j] != parent_label) continue; // restrict to parent
                    float w = G_global->weights[(size_t)q];
                    if (w <= 0.0f) continue;
                    Cand_indices_global_local.push_back(j);
                    Cand_w_local.push_back(w);
                    ++nnz_cand;
                }
            }
            Cand_indptr_local[(size_t)i + 1] = nnz_cand;
        }
    }

    // Gatherv helper for int64 vectors
    static void gatherv_i64(MPI_Comm comm, int rank, const std::vector<int64_t>& send,
        const std::vector<int>& counts, const std::vector<int>& displs,
        std::vector<int64_t>& recv)
    {
        mpi_check(MPI_Gatherv((void*)send.data(), (int)send.size(), MPI_LONG_LONG,
            rank == 0 ? (void*)recv.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_LONG_LONG, 0, comm),
            "Gatherv(i64)");
    }

    // Gatherv helper for int32
    static void gatherv_i32(MPI_Comm comm, int rank, const std::vector<int32_t>& send,
        const std::vector<int>& counts, const std::vector<int>& displs,
        std::vector<int32_t>& recv)
    {
        mpi_check(MPI_Gatherv((void*)send.data(), (int)send.size(), MPI_INT,
            rank == 0 ? (void*)recv.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_INT, 0, comm),
            "Gatherv(i32)");
    }

    // Gatherv helper for float
    static void gatherv_f32(MPI_Comm comm, int rank, const std::vector<float>& send,
        const std::vector<int>& counts, const std::vector<int>& displs,
        std::vector<float>& recv)
    {
        mpi_check(MPI_Gatherv((void*)send.data(), (int)send.size(), MPI_FLOAT,
            rank == 0 ? (void*)recv.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_FLOAT, 0, comm),
            "Gatherv(f32)");
    }

    // Rank0: arctan centering (per gene) on CSR values. Zeros remain zeros.
    static void arctan_center_inplace(int32_t m, int64_t n_cells,
        const std::vector<int64_t>& indptr,
        const std::vector<int32_t>& indices,
        std::vector<float>& data)
    {
        std::vector<std::vector<double>> vals((size_t)m);
        vals.reserve((size_t)m);

        // collect values per gene
        const int64_t nnz = (int64_t)data.size();
        for (int64_t p = 0; p < nnz; ++p) {
            const int32_t g = indices[(size_t)p];
            if (g < 0 || g >= m) continue;
            vals[(size_t)g].push_back((double)data[(size_t)p]);
        }

        std::vector<double> expM((size_t)m, 0.0);

        for (int32_t g = 0; g < m; ++g) {
            auto& v = vals[(size_t)g];
            if (v.empty()) continue;
            std::sort(v.begin(), v.end(), std::greater<double>());

            double S = 0.0;
            for (double x : v) S += x;
            if (!(S > 0.0)) continue;

            const double th = 0.8 * S;
            const int64_t vsz = (int64_t)v.size();

            int64_t binCellNum = (int64_t)(n_cells / 1000);
            int64_t j_end = 1;

            if (binCellNum <= 9) {
                double cs = 0.0;
                j_end = 0;
                for (int64_t i = 0; i < vsz; ++i) {
                    cs += v[(size_t)i];
                    if (cs > th) { j_end = i + 1; break; }
                }
                if (j_end <= 0) j_end = vsz;
            }
            else {
                int64_t loopNum = (n_cells - binCellNum) / binCellNum;
                if (loopNum <= 0) {
                    double cs = 0.0;
                    for (int64_t i = 0; i < vsz; ++i) {
                        cs += v[(size_t)i];
                        if (cs > th) { j_end = i + 1; break; }
                    }
                }
                else {
                    double cs = 0.0;
                    int64_t idx_hit = -1;
                    for (int64_t t = 0; t < loopNum; ++t) {
                        int64_t end = (t + 1) * binCellNum;
                        if (end > vsz) end = vsz;
                        // compute cs at end incrementally
                        // (for simplicity: accumulate until end)
                        // NOTE: this is exact but O(vsz). For m<=450 OK.
                        cs = 0.0;
                        for (int64_t i = 0; i < end; ++i) cs += v[(size_t)i];
                        if (cs > th) { idx_hit = end; break; }
                    }
                    if (idx_hit > 0) j_end = idx_hit;
                    else j_end = (int64_t)std::min<int64_t>(vsz, loopNum * binCellNum);
                }
            }

            if (j_end < 1) j_end = 1;
            // mean over first j_end values (zeros if j_end > nnz)
            if (j_end <= vsz) {
                double cs = 0.0;
                for (int64_t i = 0; i < j_end; ++i) cs += v[(size_t)i];
                expM[(size_t)g] = cs / (double)j_end;
            }
            else {
                expM[(size_t)g] = S / (double)j_end;
            }
        }

        // apply transform to nonzeros
        (void)indptr;
        for (size_t p = 0; p < data.size(); ++p) {
            int32_t g = indices[p];
            double x = (double)data[p];
            double em = expM[(size_t)g];
            double y = 10.0 * (std::atan(x - em) + std::atan(em));
            data[p] = (float)y;
        }
    }

    // Rank0: zscore per gene (zeros included in mean/var), but only nonzeros are modified; zeros stay zeros.
    static void zscore_inplace(int32_t m, int64_t n_cells,
        const std::vector<int64_t>& indptr,
        const std::vector<int32_t>& indices,
        std::vector<float>& data)
    {
        std::vector<double> sums((size_t)m, 0.0), sums2((size_t)m, 0.0);

        for (size_t p = 0; p < data.size(); ++p) {
            int32_t g = indices[p];
            double x = (double)data[p];
            sums[(size_t)g] += x;
            sums2[(size_t)g] += x * x;
        }

        std::vector<double> mu((size_t)m, 0.0), inv_std((size_t)m, 0.0);
        const double invN = 1.0 / (double)std::max<int64_t>(1, n_cells);

        for (int32_t g = 0; g < m; ++g) {
            mu[(size_t)g] = sums[(size_t)g] * invN;
            double var = (sums2[(size_t)g] * invN) - mu[(size_t)g] * mu[(size_t)g];
            if (var < 0.0) var = 0.0;
            double sd = std::sqrt(var);
            inv_std[(size_t)g] = (sd > 0.0) ? (1.0 / sd) : 0.0;
        }

        (void)indptr;
        for (size_t p = 0; p < data.size(); ++p) {
            int32_t g = indices[p];
            double sd_inv = inv_std[(size_t)g];
            if (sd_inv == 0.0) { data[p] = 0.0f; continue; }
            double x = (double)data[p];
            x = (x - mu[(size_t)g]) * sd_inv;
            data[p] = (float)x;
        }

        // prune explicit zeros (optional)
        // We keep them; later Gram ignores exact zeros anyway.
    }

    // Multiply symmetric matrix (m×m) by vector (m)
    static void sym_matvec(const std::vector<double>& A, int32_t m,
        const std::vector<double>& x, std::vector<double>& y)
    {
        y.assign((size_t)m, 0.0);
        for (int32_t i = 0; i < m; ++i) {
            const double* row = &A[(size_t)i * (size_t)m];
            double s = 0.0;
            for (int32_t j = 0; j < m; ++j) s += row[j] * x[(size_t)j];
            y[(size_t)i] = s;
        }
    }

    // Top eigenvectors by power iteration + Gram-Schmidt
    static std::vector<double> top_eigenvectors_power(const std::vector<double>& Gram,
        int32_t m, int32_t d,
        int iters, int seed)
    {
        std::vector<double> V((size_t)m * (size_t)d, 0.0); // col-major: V[g + k*m]
        std::vector<double> v((size_t)m), w((size_t)m);

        std::mt19937 rng((uint32_t)seed);
        std::normal_distribution<double> nd(0.0, 1.0);

        for (int32_t k = 0; k < d; ++k) {
            for (int32_t i = 0; i < m; ++i) v[(size_t)i] = nd(rng);

            // normalize
            auto norm2 = [&]() {
                double s = 0.0; for (double a : v) s += a * a; return std::sqrt(std::max(1e-30, s));
                };
            double nv = norm2();
            for (double& a : v) a /= nv;

            for (int t = 0; t < iters; ++t) {
                sym_matvec(Gram, m, v, w);

                // orthogonalize vs previous eigenvectors
                for (int32_t j = 0; j < k; ++j) {
                    double dot = 0.0;
                    for (int32_t i = 0; i < m; ++i) dot += w[(size_t)i] * V[(size_t)i + (size_t)j * (size_t)m];
                    for (int32_t i = 0; i < m; ++i) w[(size_t)i] -= dot * V[(size_t)i + (size_t)j * (size_t)m];
                }

                double nw = 0.0;
                for (double a : w) nw += a * a;
                nw = std::sqrt(std::max(1e-30, nw));
                for (int32_t i = 0; i < m; ++i) v[(size_t)i] = w[(size_t)i] / nw;
            }

            // save
            for (int32_t i = 0; i < m; ++i) V[(size_t)i + (size_t)k * (size_t)m] = v[(size_t)i];
        }
        return V;
    }

    // Gram = X^T X from sparse CSR X (rows=cells, cols=m)
    static std::vector<double> compute_gram(int32_t m,
        const std::vector<int64_t>& indptr,
        const std::vector<int32_t>& indices,
        const std::vector<float>& data)
    {
        std::vector<double> G((size_t)m * (size_t)m, 0.0);

        const int64_t n = (int64_t)indptr.size() - 1;
        for (int64_t i = 0; i < n; ++i) {
            int64_t s = indptr[(size_t)i];
            int64_t e = indptr[(size_t)i + 1];
            for (int64_t a = s; a < e; ++a) {
                int32_t ca = indices[(size_t)a];
                double va = (double)data[(size_t)a];
                for (int64_t b = s; b < e; ++b) {
                    int32_t cb = indices[(size_t)b];
                    double vb = (double)data[(size_t)b];
                    G[(size_t)ca * (size_t)m + (size_t)cb] += va * vb;
                }
            }
        }
        return G;
    }

    // Z = X * V (X sparse CSR n×m, V col-major m×d) -> dense n×d, then L2 normalize
    static std::vector<float> embed_and_l2norm(int32_t m, int32_t d,
        const std::vector<int64_t>& indptr,
        const std::vector<int32_t>& indices,
        const std::vector<float>& data,
        const std::vector<double>& V)
    {
        const int64_t n = (int64_t)indptr.size() - 1;
        std::vector<float> Z((size_t)n * (size_t)d, 0.0f);

        for (int64_t i = 0; i < n; ++i) {
            float* zi = &Z[(size_t)i * (size_t)d];
            int64_t s = indptr[(size_t)i];
            int64_t e = indptr[(size_t)i + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t g = indices[(size_t)p];
                float x = data[(size_t)p];
                const double* vg = &V[(size_t)g];
                for (int32_t k = 0; k < d; ++k) {
                    zi[(size_t)k] += x * (float)vg[(size_t)k * (size_t)m];
                }
            }
            double n2 = 0.0;
            for (int32_t k = 0; k < d; ++k) n2 += (double)zi[(size_t)k] * (double)zi[(size_t)k];
            double inv = (n2 > 0.0) ? (1.0 / std::sqrt(n2)) : 0.0;
            for (int32_t k = 0; k < d; ++k) zi[(size_t)k] = (float)(zi[(size_t)k] * inv);
        }
        return Z;
    }

    // Build fused cosine graph from candidate edges (candidate neighbors per row).
    // candidates are given with neighbor global ids and global weights; we map to local ids before calling this.
    struct CSRGraphF32 {
        int32_t n = 0;
        std::vector<int64_t> indptr;
        std::vector<int32_t> indices;
        std::vector<float>   weights;
    };

    static CSRGraphF32 build_cosine_graph_from_candidates(
        const std::vector<float>& Z, int32_t n, int32_t d,
        const std::vector<int64_t>& cand_indptr,
        const std::vector<int32_t>& cand_j,           // local neighbor ids
        const std::vector<float>& cand_w_global,    // global similarity weights
        int k_knn,
        double mix_alpha)
    {
        CSRGraphF32 A;
        A.n = n;

        // adjacency as maps to enforce symmetry-max cheaply
        std::vector<std::unordered_map<int32_t, float>> adj((size_t)n);
        for (int32_t i = 0; i < n; ++i) adj[(size_t)i].reserve((size_t)k_knn * 2);

        const float alpha = (float)mix_alpha;
        const float beta = (float)(1.0 - mix_alpha);

        // for each row: compute cosine sim for candidates, take top-k, then fuse with global weights
        for (int32_t i = 0; i < n; ++i) {
            const int64_t s = cand_indptr[(size_t)i];
            const int64_t e = cand_indptr[(size_t)i + 1];
            if (s == e) continue;

            // compute sim on candidates
            // store (sim, j, global_w)
            struct Cand { float sim; int32_t j; float gw; };
            std::vector<Cand> tmp;
            tmp.reserve((size_t)(e - s));

            const float* zi = &Z[(size_t)i * (size_t)d];
            for (int64_t p = s; p < e; ++p) {
                const int32_t j = cand_j[(size_t)p];
                if (j == i) continue;
                const float* zj = &Z[(size_t)j * (size_t)d];
                float dot = 0.0f;
                for (int32_t t = 0; t < d; ++t) dot += zi[(size_t)t] * zj[(size_t)t];
                if (dot < 0.0f) dot = 0.0f; // match python np.maximum(A_loc.data,0)
                tmp.push_back(Cand{ dot, j, cand_w_global[(size_t)p] });
            }
            if (tmp.empty()) continue;

            // top-k by cosine sim
            const int kk = std::min(k_knn, (int)tmp.size());
            std::nth_element(tmp.begin(), tmp.end() - kk, tmp.end(),
                [](const Cand& a, const Cand& b) { return a.sim < b.sim; });
            // keep last kk
            std::vector<Cand> top(tmp.end() - kk, tmp.end());
            std::sort(top.begin(), top.end(),
                [](const Cand& a, const Cand& b) { return a.sim > b.sim; });

            // start with (1-alpha)*global weights for all candidates (keeps A_global[C,C])
            for (const auto& c : tmp) {
                if (c.gw <= 0.0f) continue;
                float w = beta * c.gw;
                if (w <= 0.0f) continue;
                adj[(size_t)i][c.j] = std::max(adj[(size_t)i][c.j], w);
            }
            // add alpha*cosine for selected top-k
            for (const auto& c : top) {
                if (c.sim <= 0.0f) continue;
                float add = alpha * c.sim;
                auto it = adj[(size_t)i].find(c.j);
                if (it == adj[(size_t)i].end()) adj[(size_t)i][c.j] = add;
                else it->second += add;
            }
        }

        // Symmetrize by max: adj[i][j] and adj[j][i] -> both set to max
        for (int32_t i = 0; i < n; ++i) {
            for (auto& kv : adj[(size_t)i]) {
                const int32_t j = kv.first;
                if (j == i) continue;
                float wij = kv.second;
                float wji = 0.0f;
                auto it = adj[(size_t)j].find(i);
                if (it != adj[(size_t)j].end()) wji = it->second;
                float mmax = std::max(wij, wji);
                kv.second = mmax;
                adj[(size_t)j][i] = mmax;
            }
            adj[(size_t)i].erase(i);
        }

        // Row-normalize
        for (int32_t i = 0; i < n; ++i) {
            double ssum = 0.0;
            for (auto& kv : adj[(size_t)i]) ssum += (double)kv.second;
            if (ssum <= 0.0) continue;
            const float inv = (float)(1.0 / ssum);
            for (auto& kv : adj[(size_t)i]) kv.second *= inv;
        }

        // Export CSR
        A.indptr.assign((size_t)n + 1, 0);
        int64_t nnz = 0;
        for (int32_t i = 0; i < n; ++i) {
            A.indptr[(size_t)i] = nnz;
            nnz += (int64_t)adj[(size_t)i].size();
        }
        A.indptr[(size_t)n] = nnz;
        A.indices.resize((size_t)nnz);
        A.weights.resize((size_t)nnz);

        int64_t pos = 0;
        for (int32_t i = 0; i < n; ++i) {
            // deterministic order
            std::vector<std::pair<int32_t, float>> row;
            row.reserve(adj[(size_t)i].size());
            for (auto& kv : adj[(size_t)i]) row.push_back(kv);
            std::sort(row.begin(), row.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

            for (auto& pr : row) {
                A.indices[(size_t)pos] = pr.first;
                A.weights[(size_t)pos] = pr.second;
                ++pos;
            }
        }

        return A;
    }

    // Leiden on rank0 for a symmetric similarity CSR graph (n×n). Returns membership.
    static std::vector<int32_t> leiden_local_similarity(const CSRGraphF32& A,
        double resolution,
        double beta,
        int n_iterations,
        int seed,
        bool verbose)
    {
        const int32_t n = A.n;
        std::vector<int32_t> labs((size_t)n, 0);
        if (n <= 0) return labs;

        // Build upper triangle edge list
        std::vector<int64_t> src, dst;
        std::vector<double>  w;
        src.reserve((size_t)n * 20);
        dst.reserve((size_t)n * 20);
        w.reserve((size_t)n * 20);

        double sum_w = 0.0;
        std::vector<double> strength((size_t)n, 0.0);

        for (int32_t i = 0; i < n; ++i) {
            int64_t s = A.indptr[(size_t)i];
            int64_t e = A.indptr[(size_t)i + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t j = A.indices[(size_t)p];
                double ww = (double)A.weights[(size_t)p];
                if (j <= i) continue; // upper only
                if (!(ww > 0.0)) continue;
                src.push_back(i);
                dst.push_back(j);
                w.push_back(ww);
                strength[(size_t)i] += ww;
                strength[(size_t)j] += ww;
                sum_w += ww;
            }
        }

        if (!(sum_w > 0.0) || src.empty()) {
            // all singletons
            for (int32_t i = 0; i < n; ++i) labs[(size_t)i] = i;
            return labs;
        }

        // IMPORTANT: scaled gamma to avoid singleton explosion with node weights
        const double gamma_scaled = resolution / (2.0 * sum_w);

        const igraph_int_t nV = (igraph_int_t)n;
        const igraph_int_t mE = (igraph_int_t)src.size();

        igraph_vector_int_t edges;
        igraph_vector_int_init(&edges, 2 * mE);
        igraph_vector_t wvec;
        igraph_vector_init(&wvec, mE);

        for (igraph_int_t e = 0; e < mE; ++e) {
            VECTOR(edges)[2 * e + 0] = (igraph_int_t)src[(size_t)e];
            VECTOR(edges)[2 * e + 1] = (igraph_int_t)dst[(size_t)e];
            VECTOR(wvec)[e] = (igraph_real_t)w[(size_t)e];
        }

        igraph_t g;
        igraph_check(igraph_create(&g, &edges, nV, /*directed=*/0), "igraph_create");

        igraph_vector_t node_w;
        igraph_vector_init(&node_w, nV);
        for (igraph_int_t i = 0; i < nV; ++i) VECTOR(node_w)[i] = (igraph_real_t)strength[(size_t)i];

        igraph_vector_int_t membership;
        igraph_vector_int_init(&membership, nV);
        for (igraph_int_t i = 0; i < nV; ++i) VECTOR(membership)[i] = i;

        igraph_rng_seed(igraph_rng_default(), (igraph_uint_t)seed);

        igraph_int_t nb_clusters = 0;
        igraph_real_t quality = 0.0;

#if defined(IGRAPH_VERSION_MAJOR) && (IGRAPH_VERSION_MAJOR >= 1)
        igraph_check(
            igraph_community_leiden(
                &g,
                &wvec,
                &node_w,
                /*vertex_in_weights=*/nullptr,
                (igraph_real_t)gamma_scaled,
                (igraph_real_t)beta,
                /*start=*/0,
                (igraph_int_t)n_iterations,
                &membership,
                &nb_clusters,
                &quality
            ),
            "igraph_community_leiden(v1+)"
        );
#else
        igraph_check(
            igraph_community_leiden(
                &g,
                &wvec,
                &node_w,
                (igraph_real_t)gamma_scaled,
                (igraph_real_t)beta,
                /*start=*/0,
                (igraph_integer_t)n_iterations,
                &membership,
                &nb_clusters,
                &quality
            ),
            "igraph_community_leiden(0.10.x)"
        );
#endif

        if (verbose) {
            std::cout << "[rank 0] refine Leiden: n=" << n
                << " edges=" << (long long)mE
                << " clusters=" << (long long)nb_clusters
                << " quality=" << (double)quality
                << " resolution=" << resolution
                << " gamma_scaled=" << gamma_scaled
                << "\n";
        }

        for (igraph_int_t i = 0; i < nV; ++i) labs[(size_t)i] = (int32_t)VECTOR(membership)[i];

        igraph_vector_int_destroy(&membership);
        igraph_vector_destroy(&node_w);
        igraph_destroy(&g);
        igraph_vector_destroy(&wvec);
        igraph_vector_int_destroy(&edges);

        return labs;
    }

    // -------------------- main refinement --------------------

    std::vector<int32_t> refine_pca_cosine_mpi(
        MPI_Comm comm,
        const CSRMatrixF32& X_local,
        const std::vector<std::string>& gene_names,
        const std::vector<int32_t>& labels_major_local,
        const std::vector<std::string>& refine_gene_set,
        const KNNGraphLocal* G_global,
        const RefineParams& p)
    {
        int rank = 0, world = 1;
        mpi_check(MPI_Comm_rank(comm, &rank), "Comm_rank");
        mpi_check(MPI_Comm_size(comm, &world), "Comm_size");

        const int64_t local_rows = (int64_t)labels_major_local.size();

        // Compute row0 via Exscan for safety (do not assume caller partition is perfect)
        long long local_ll = (long long)local_rows;
        long long row0_ll = 0;
        mpi_check(MPI_Exscan(&local_ll, &row0_ll, 1, MPI_LONG_LONG, MPI_SUM, comm), "Exscan(row0)");
        if (rank == 0) row0_ll = 0;
        long long N_ll = 0;
        mpi_check(MPI_Allreduce(&local_ll, &N_ll, 1, MPI_LONG_LONG, MPI_SUM, comm), "Allreduce(N)");
        const int64_t row0_global = (int64_t)row0_ll;
        const int64_t N_global = (int64_t)N_ll;

        // All ranks need labels_major_global to filter candidate neighbors by parent cluster
        std::vector<int32_t> labels_major_global = allgather_labels(comm, labels_major_local,
            local_rows, row0_global, N_global);

        // Map refine genes -> selection indices
        std::vector<int32_t> sel_gene_idx;
        std::vector<int32_t> gene_to_selpos;
        map_refine_genes(gene_names, refine_gene_set, sel_gene_idx, gene_to_selpos);
        const int32_t m = (int32_t)sel_gene_idx.size();

        if (rank == 0 && p.verbose) {
            std::cout << "[rank 0] refine: gene_set size=" << refine_gene_set.size()
                << " mapped=" << m << "\n";
        }
        if (m <= 1) {
            // nothing to refine
            return labels_major_local;
        }

        // Collect unique parent labels on rank0
        std::vector<int32_t> parents_local = labels_major_local;
        std::sort(parents_local.begin(), parents_local.end());
        parents_local.erase(std::unique(parents_local.begin(), parents_local.end()), parents_local.end());

        // gather uniques to rank0
        int local_u = (int)parents_local.size();
        std::vector<int> uc_counts, uc_displs;
        if (rank == 0) { uc_counts.resize(world); uc_displs.resize(world); }
        mpi_check(MPI_Gather(&local_u, 1, MPI_INT,
            rank == 0 ? uc_counts.data() : nullptr, 1, MPI_INT,
            0, comm),
            "Gather(unique counts)");

        std::vector<int32_t> parents_all;
        if (rank == 0) {
            uc_displs[0] = 0;
            for (int r = 1; r < world; ++r) uc_displs[r] = uc_displs[r - 1] + uc_counts[r - 1];
            int tot = uc_displs[world - 1] + uc_counts[world - 1];
            parents_all.resize((size_t)tot);
        }

        mpi_check(MPI_Gatherv(parents_local.data(), local_u, MPI_INT,
            rank == 0 ? parents_all.data() : nullptr,
            rank == 0 ? uc_counts.data() : nullptr,
            rank == 0 ? uc_displs.data() : nullptr,
            MPI_INT, 0, comm),
            "Gatherv(parents)");

        std::vector<int32_t> parents;
        if (rank == 0) {
            std::sort(parents_all.begin(), parents_all.end());
            parents_all.erase(std::unique(parents_all.begin(), parents_all.end()), parents_all.end());
            parents = parents_all;
        }

        // Broadcast parent list
        int n_par = 0;
        if (rank == 0) n_par = (int)parents.size();
        mpi_check(MPI_Bcast(&n_par, 1, MPI_INT, 0, comm), "Bcast(n_par)");
        if (rank != 0) parents.resize((size_t)n_par);
        mpi_check(MPI_Bcast(parents.data(), n_par, MPI_INT, 0, comm), "Bcast(parents)");

        // Rank0 builds refined global labels, then scatter back
        std::vector<int32_t> labels_refined_global;
        std::vector<int32_t> labels_refined_local = labels_major_local;

        // Gather full labels to rank0 (contiguous by rank)
        std::vector<int> row_counts, row_displs;
        row_counts.resize(world);
        row_displs.resize(world);
        int local_n = (int)local_rows;

        mpi_check(MPI_Allgather(&local_n, 1, MPI_INT, row_counts.data(), 1, MPI_INT, comm),
            "Allgather(row_counts)");

        row_displs[0] = 0;
        for (int r = 1; r < world; ++r) row_displs[r] = row_displs[r - 1] + row_counts[r - 1];
        const int totalN = row_displs[world - 1] + row_counts[world - 1];
        if ((int64_t)totalN != N_global) throw std::runtime_error("refine: row partition mismatch");

        if (rank == 0) {
            labels_refined_global.resize((size_t)N_global);
        }
        mpi_check(MPI_Gatherv((void*)labels_major_local.data(), local_n, MPI_INT,
            rank == 0 ? (void*)labels_refined_global.data() : nullptr,
            row_counts.data(), row_displs.data(), MPI_INT,
            0, comm),
            "Gatherv(labels_major)");

        int32_t next_label = 0;
        if (rank == 0) {
            int32_t mx = 0;
            for (auto v : labels_refined_global) mx = std::max(mx, v);
            next_label = mx + 1;
        }

        // Process each parent cluster
        for (int pi = 0; pi < n_par; ++pi) {
            const int32_t parent = parents[(size_t)pi];

            // ---- build local cluster pack ----
            std::vector<int64_t> cell_ids_loc;
            std::vector<int64_t> X_indptr_loc;
            std::vector<int32_t> X_idx_loc;
            std::vector<float>   X_dat_loc;

            std::vector<int64_t> C_indptr_loc;
            std::vector<int64_t> C_jg_loc;
            std::vector<float>   C_w_loc;

            // Partition check for global candidate graph compatibility
            if (p.use_global_candidates && !G_global) {
                if (rank == 0) {
                    std::cout << "[rank 0] refine WARNING: use_global_candidates=true but G_global is null; skipping parent.\n";
                }
                continue;
            }

            build_local_cluster_csr_and_candidates(
                X_local, labels_major_local, gene_to_selpos,
                parent,
                G_global, labels_major_global,
                row0_global,
                cell_ids_loc,
                X_indptr_loc, X_idx_loc, X_dat_loc,
                C_indptr_loc, C_jg_loc, C_w_loc
            );

            const int n_loc = (int)cell_ids_loc.size();
            const int nnz_x_loc = (int)X_dat_loc.size();
            const int nnz_c_loc = (int)C_w_loc.size();

            // gather counts to rank0
            std::vector<int> n_counts, n_displs;
            std::vector<int> x_counts, x_displs;
            std::vector<int> c_counts, c_displs;

            if (rank == 0) {
                n_counts.resize(world); n_displs.resize(world);
                x_counts.resize(world); x_displs.resize(world);
                c_counts.resize(world); c_displs.resize(world);
            }

            mpi_check(MPI_Gather(&n_loc, 1, MPI_INT, rank == 0 ? n_counts.data() : nullptr, 1, MPI_INT, 0, comm), "Gather(n_loc)");
            mpi_check(MPI_Gather(&nnz_x_loc, 1, MPI_INT, rank == 0 ? x_counts.data() : nullptr, 1, MPI_INT, 0, comm), "Gather(nnz_x)");
            mpi_check(MPI_Gather(&nnz_c_loc, 1, MPI_INT, rank == 0 ? c_counts.data() : nullptr, 1, MPI_INT, 0, comm), "Gather(nnz_c)");

            int n_tot = 0, x_tot = 0, c_tot = 0;
            if (rank == 0) {
                n_displs[0] = 0; x_displs[0] = 0; c_displs[0] = 0;
                for (int r = 1; r < world; ++r) {
                    n_displs[r] = n_displs[r - 1] + n_counts[r - 1];
                    x_displs[r] = x_displs[r - 1] + x_counts[r - 1];
                    c_displs[r] = c_displs[r - 1] + c_counts[r - 1];
                }
                n_tot = n_displs[world - 1] + n_counts[world - 1];
                x_tot = x_displs[world - 1] + x_counts[world - 1];
                c_tot = c_displs[world - 1] + c_counts[world - 1];
            }

            // Gather cell ids
            std::vector<int64_t> cell_ids_all;
            if (rank == 0) cell_ids_all.resize((size_t)n_tot);
            gatherv_i64(comm, rank, cell_ids_loc, n_counts, n_displs, cell_ids_all);

            // Gather X indptr (len n_loc+1 each)
            std::vector<int> ip_counts, ip_displs;
            int ip_loc = n_loc + 1;
            if (rank == 0) { ip_counts.resize(world); ip_displs.resize(world); }
            mpi_check(MPI_Gather(&ip_loc, 1, MPI_INT, rank == 0 ? ip_counts.data() : nullptr, 1, MPI_INT, 0, comm), "Gather(ip_loc)");
            int ip_tot = 0;
            if (rank == 0) {
                ip_displs[0] = 0;
                for (int r = 1; r < world; ++r) ip_displs[r] = ip_displs[r - 1] + ip_counts[r - 1];
                ip_tot = ip_displs[world - 1] + ip_counts[world - 1];
            }
            std::vector<int64_t> X_indptr_all;
            if (rank == 0) X_indptr_all.resize((size_t)ip_tot);
            gatherv_i64(comm, rank, X_indptr_loc, ip_counts, ip_displs, X_indptr_all);

            // Gather X indices and data
            std::vector<int32_t> X_idx_all;
            std::vector<float>   X_dat_all;
            if (rank == 0) { X_idx_all.resize((size_t)x_tot); X_dat_all.resize((size_t)x_tot); }
            gatherv_i32(comm, rank, X_idx_loc, x_counts, x_displs, X_idx_all);
            gatherv_f32(comm, rank, X_dat_loc, x_counts, x_displs, X_dat_all);

            // Gather candidate indptr (len n_loc+1)
            std::vector<int64_t> C_indptr_all;
            if (rank == 0) C_indptr_all.resize((size_t)ip_tot);
            gatherv_i64(comm, rank, C_indptr_loc, ip_counts, ip_displs, C_indptr_all);

            // Gather candidate neighbor global ids + weights
            std::vector<int64_t> C_jg_all;
            std::vector<float>   C_w_all;
            if (rank == 0) { C_jg_all.resize((size_t)c_tot); C_w_all.resize((size_t)c_tot); }
            gatherv_i64(comm, rank, C_jg_loc, c_counts, c_displs, C_jg_all);
            gatherv_f32(comm, rank, C_w_loc, c_counts, c_displs, C_w_all);

            // ---- rank0 computes refinement for this parent ----
            if (rank == 0) {
                const int64_t n_c = (int64_t)n_tot;
                if (n_c < 2) continue;

                if (p.verbose) {
                    std::cout << "[rank 0] refine parent=" << parent
                        << " n_c=" << n_c
                        << " nnz_x=" << x_tot
                        << " nnz_cand=" << c_tot
                        << "\n";
                }

                // Rebuild CSR indptr for X (because gathered indptr segments restart at 0)
                std::vector<int64_t> X_indptr((size_t)n_c + 1, 0);
                int64_t row_cur = 0;
                int64_t nnz_cur = 0;

                // iterate ranks to reconstruct row nnz from each local indptr segment
                int64_t ip_off = 0;
                int64_t x_off = 0;
                for (int r = 0; r < world; ++r) {
                    const int nr = n_counts[r];
                    const int ipr = ip_counts[r]; // nr+1
                    if (ipr != nr + 1) throw std::runtime_error("refine: bad indptr counts");
                    const int64_t* ip = &X_indptr_all[(size_t)ip_off];
                    // ip[0] should be 0; ip[nr] should be x_counts[r]
                    for (int i = 0; i < nr; ++i) {
                        int64_t rn = ip[(size_t)i + 1] - ip[(size_t)i];
                        X_indptr[(size_t)row_cur + 1] = X_indptr[(size_t)row_cur] + rn;
                        ++row_cur;
                    }
                    ip_off += ipr;
                    x_off += x_counts[r];
                }
                if (row_cur != n_c) throw std::runtime_error("refine: row_cur mismatch");
                if (X_indptr.back() != (int64_t)X_dat_all.size()) {
                    // tolerate but warn
                    if (p.verbose) std::cout << "[rank 0] refine warning: X_indptr nnz mismatch\n";
                }

                // Similarly rebuild candidate indptr
                std::vector<int64_t> C_indptr((size_t)n_c + 1, 0);
                row_cur = 0;
                for (int r = 0; r < world; ++r) {
                    const int nr = n_counts[r];
                    const int64_t* ip = &C_indptr_all[(size_t)(std::accumulate(ip_counts.begin(), ip_counts.begin() + r, 0))];
                    // NOTE: above compute is clumsy; do incremental like before:
                }
                // rebuild candidate indptr properly:
                {
                    int64_t ip2_off = 0;
                    for (int r = 0; r < world; ++r) {
                        const int nr = n_counts[r];
                        const int ipr = ip_counts[r];
                        const int64_t* ip = &C_indptr_all[(size_t)ip2_off];
                        for (int i = 0; i < nr; ++i) {
                            int64_t rn = ip[(size_t)i + 1] - ip[(size_t)i];
                            C_indptr[(size_t)row_cur + 1] = C_indptr[(size_t)row_cur] + rn;
                            ++row_cur;
                        }
                        ip2_off += ipr;
                    }
                }
                if (C_indptr.back() != (int64_t)C_w_all.size()) {
                    if (p.verbose) std::cout << "[rank 0] refine warning: Cand indptr nnz mismatch\n";
                }

                // Build global->local mapping for cluster cells
                std::unordered_map<int64_t, int32_t> g2l;
                g2l.reserve((size_t)n_c * 2);
                for (int32_t i = 0; i < (int32_t)n_c; ++i) g2l[cell_ids_all[(size_t)i]] = i;

                // Map candidate neighbor global ids -> local ids, drop those not in cluster
                std::vector<int64_t> C_indptr2((size_t)n_c + 1, 0);
                std::vector<int32_t> C_j_local;
                std::vector<float>   C_w2;

                C_j_local.reserve((size_t)C_w_all.size());
                C_w2.reserve((size_t)C_w_all.size());

                for (int32_t i = 0; i < (int32_t)n_c; ++i) {
                    int64_t s = C_indptr[(size_t)i];
                    int64_t e = C_indptr[(size_t)i + 1];
                    for (int64_t p2 = s; p2 < e; ++p2) {
                        int64_t jg = C_jg_all[(size_t)p2];
                        auto it = g2l.find(jg);
                        if (it == g2l.end()) continue;
                        int32_t jl = it->second;
                        if (jl == i) continue;
                        float w = C_w_all[(size_t)p2];
                        if (w <= 0.0f) continue;
                        C_j_local.push_back(jl);
                        C_w2.push_back(w);
                    }
                    C_indptr2[(size_t)i + 1] = (int64_t)C_j_local.size();
                }

                // ---- optional arctan ----
                if (p.use_arctan) {
                    arctan_center_inplace(m, n_c, X_indptr, X_idx_all, X_dat_all);
                }

                // ---- zscore ----
                zscore_inplace(m, n_c, X_indptr, X_idx_all, X_dat_all);

                // ---- PCA ----
                // Gram = X^T X
                auto Gram = compute_gram(m, X_indptr, X_idx_all, X_dat_all);
                const int d_use = std::min(p.n_pcs, m);
                auto V = top_eigenvectors_power(Gram, m, d_use, p.power_iters, p.seed);

                // Z = X V, L2 normalize
                auto Z = embed_and_l2norm(m, d_use, X_indptr, X_idx_all, X_dat_all, V);

                // ---- build cosine+mix graph and run Leiden ----
                auto A = build_cosine_graph_from_candidates(
                    Z, (int32_t)n_c, (int32_t)d_use,
                    C_indptr2, C_j_local, C_w2,
                    p.k_knn, p.mix_alpha
                );

                auto child = leiden_local_similarity(A, p.resolution, p.beta, p.n_iterations, p.seed, p.verbose);

                // Decide which children to accept: size thresholds, keep parent for largest child
                std::unordered_map<int32_t, int32_t> sz;
                sz.reserve(64);
                for (int32_t i = 0; i < (int32_t)n_c; ++i) sz[child[(size_t)i]]++;

                // if no split, continue
                if (sz.size() <= 1) continue;

                // largest child stays parent label
                int32_t child_keep = sz.begin()->first;
                int32_t best = -1;
                for (auto& kv : sz) {
                    if (kv.second > best) { best = kv.second; child_keep = kv.first; }
                }

                const int min_sz = std::max(p.min_child_size_abs,
                    (int)std::ceil(p.min_child_size_frac_parent * (double)n_c));

                // Assign new ids to accepted children (except the kept one)
                std::unordered_map<int32_t, int32_t> child2new;
                child2new.reserve(sz.size());
                child2new[child_keep] = parent;

                for (auto& kv : sz) {
                    const int32_t c = kv.first;
                    const int32_t s = kv.second;
                    if (c == child_keep) continue;
                    if (s < min_sz) {
                        // too small -> merge back to parent
                        child2new[c] = parent;
                    }
                    else {
                        child2new[c] = next_label++;
                    }
                }

                // Update refined global labels
                for (int32_t i = 0; i < (int32_t)n_c; ++i) {
                    int64_t gid = cell_ids_all[(size_t)i];
                    int32_t newlab = child2new[child[(size_t)i]];
                    labels_refined_global[(size_t)gid] = newlab;
                }
            }

            // sync ranks between parents (optional)
            //mpi_check(MPI_Barrier(comm), "Barrier(parent)");
        }

        // Scatter refined global labels back to ranks
        if (rank == 0 && (int)labels_refined_global.size() != totalN) {
            throw std::runtime_error("refine: labels_refined_global size mismatch.");
        }

        std::vector<int32_t> refined_local((size_t)local_rows, 0);
        mpi_check(MPI_Scatterv(rank == 0 ? labels_refined_global.data() : nullptr,
            row_counts.data(), row_displs.data(), MPI_INT,
            refined_local.data(), local_n, MPI_INT,
            0, comm),
            "Scatterv(refined labels)");

        return refined_local;
    }

} // namespace rarecell
