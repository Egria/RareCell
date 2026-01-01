#include "graph/knn.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace rarecell {

    // Convert CSR (rows=cells, cols=features, binary) to CSC (cols=features)
    static void csr_to_csc_binary(const CSRMatrixU8& csr,
        std::vector<int64_t>& csc_indptr,   // size F+1
        std::vector<int32_t>& csc_indices)  // length nnz
    {
        const int64_t R = (int64_t)csr.indptr.size() - 1;
        const int64_t F = csr.n_cols;
        csc_indptr.assign((size_t)F + 1, 0);

        // Count per column
        for (int64_t r = 0; r < R; ++r) {
            int64_t s = csr.indptr[(size_t)r];
            int64_t e = csr.indptr[(size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t f = csr.indices[(size_t)p];
                ++csc_indptr[(size_t)f + 1];
            }
        }

        // Prefix sum
        for (int64_t f = 0; f < F; ++f)
            csc_indptr[(size_t)f + 1] += csc_indptr[(size_t)f];

        const int64_t nnz = csc_indptr[(size_t)F];
        csc_indices.assign((size_t)nnz, 0);

        // Fill CSC row indices (local row ids)
        std::vector<int64_t> next = csc_indptr;
        for (int64_t r = 0; r < R; ++r) {
            int64_t s = csr.indptr[(size_t)r];
            int64_t e = csr.indptr[(size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t f = csr.indices[(size_t)p];
                int64_t pos = next[(size_t)f]++;
                csc_indices[(size_t)pos] = (int32_t)r; // local row index
            }
        }
    }

    // Ring send/recv sizes (3x int64)
    static void ring_sendrecv_sizes(MPI_Comm comm,
        int right, int left,
        int64_t send_rows, int64_t send_row0, int64_t send_idx_len,
        int64_t& recv_rows, int64_t& recv_row0, int64_t& recv_idx_len)
    {
        int64_t s[3] = { send_rows, send_row0, send_idx_len };
        int64_t r[3] = { 0, 0, 0 };

        MPI_Sendrecv(s, 3, MPI_LONG_LONG, right, 901,
            r, 3, MPI_LONG_LONG, left, 901,
            comm, MPI_STATUS_IGNORE);

        recv_rows = r[0];
        recv_row0 = r[1];
        recv_idx_len = r[2];
    }

    // Ring send/recv arrays
    static void ring_sendrecv_arrays(MPI_Comm comm,
        int right, int left,
        const int64_t* send_indptr, int indptr_len,
        const int32_t* send_indices, int64_t indices_len,
        const int32_t* send_deg, int64_t rows_len,
        int64_t* recv_indptr, int recv_indptr_len,
        int32_t* recv_indices, int64_t recv_indices_len,
        int32_t* recv_deg, int64_t recv_rows_len)
    {
        if (indices_len > (int64_t)std::numeric_limits<int>::max() ||
            recv_indices_len > (int64_t)std::numeric_limits<int>::max() ||
            rows_len > (int64_t)std::numeric_limits<int>::max() ||
            recv_rows_len > (int64_t)std::numeric_limits<int>::max())
        {
            throw std::runtime_error("MPI_Sendrecv count exceeds INT_MAX; implement chunking.");
        }

        MPI_Sendrecv(send_indptr, indptr_len, MPI_LONG_LONG, right, 902,
            recv_indptr, recv_indptr_len, MPI_LONG_LONG, left, 902,
            comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(send_indices, (int)indices_len, MPI_INT, right, 903,
            recv_indices, (int)recv_indices_len, MPI_INT, left, 903,
            comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(send_deg, (int)rows_len, MPI_INT, right, 904,
            recv_deg, (int)recv_rows_len, MPI_INT, left, 904,
            comm, MPI_STATUS_IGNORE);
    }

    // tiny top-k min-heap per row
    struct TopK {
        int k;
        std::vector<std::pair<float, int64_t>> heap; // min-heap by similarity

        explicit TopK(int kk = 30) : k(kk) { heap.reserve((size_t)kk + 2); }

        static bool cmp(const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
            return a.first > b.first; // greater => min-heap
        }

        inline void consider(float s, int64_t j) {
            if (s <= 0.0f) return;
            if ((int)heap.size() < k) {
                heap.emplace_back(s, j);
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
            else if (s > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                heap.back() = { s, j };
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }

        void finalize_sorted_desc(std::vector<int64_t>& out_idx, std::vector<float>& out_w) {
            std::sort_heap(heap.begin(), heap.end(), cmp); // ascending
            std::reverse(heap.begin(), heap.end());        // descending
            for (auto& p : heap) {
                out_idx.push_back(p.second);
                out_w.push_back(p.first);
            }
            heap.clear();
        }
    };

    KNNGraphLocal build_knn_graph_jaccard_mpi(
        MPI_Comm comm,
        const BinaryPanel& panel,
        int k,
        int block_size,
        double df_cap_frac,
        bool verbose)
    {
        int rank = 0, world = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world);

        const auto& B = panel.B_local; // CSR binary: (cells_local x features)
        const int64_t local_rows = (int64_t)B.indptr.size() - 1;
        const int64_t F = B.n_cols;

        // ---- Compute global N and this-rank row0 via MPI (DO NOT trust B.row0/B.n_rows) ----
        long long local_ll = (long long)local_rows;

        long long row0_ll = 0;
        MPI_Exscan(&local_ll, &row0_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);
        if (rank == 0) row0_ll = 0;

        long long N_total_ll = 0;
        MPI_Allreduce(&local_ll, &N_total_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);

        const int64_t row0_global = (int64_t)row0_ll;
        const int64_t row1_global = row0_global + local_rows;
        const int64_t N_global = (int64_t)N_total_ll;

        // Validate feature count consistent across ranks
        int64_t F_min = 0, F_max = 0;
        MPI_Allreduce((void*)&F, &F_min, 1, MPI_LONG_LONG, MPI_MIN, comm);
        MPI_Allreduce((void*)&F, &F_max, 1, MPI_LONG_LONG, MPI_MAX, comm);
        if (F_min != F_max) {
            throw std::runtime_error("build_knn_graph_jaccard_mpi: panel feature count differs across ranks.");
        }

        KNNGraphLocal G;
        G.n_nodes = N_global;
        G.row0 = row0_global;
        G.row1 = row1_global;
        G.indptr.assign((size_t)local_rows + 1, 0);

        // If global empty or no features, return (safe for all ranks)
        if (N_global <= 0 || F <= 0) {
            return G;
        }

        // Precompute local CSC and per-row degrees (nnz per row)
        std::vector<int64_t> csc_indptr_local;  // size F+1
        std::vector<int32_t> csc_indices_local; // nnz
        csr_to_csc_binary(B, csc_indptr_local, csc_indices_local);

        std::vector<int32_t> deg_local((size_t)local_rows, 0);
        for (int64_t r = 0; r < local_rows; ++r) {
            deg_local[(size_t)r] = (int32_t)(B.indptr[(size_t)r + 1] - B.indptr[(size_t)r]);
        }

        // Zero rows: mark and skip entirely
        std::vector<uint8_t> row_is_zero((size_t)local_rows, 0);
        for (int64_t r = 0; r < local_rows; ++r)
            if (deg_local[(size_t)r] == 0) row_is_zero[(size_t)r] = 1u;

        // ---- Global DF (feature frequency) ----
        std::vector<int64_t> df_local((size_t)F, 0);
        for (int64_t f = 0; f < F; ++f) {
            df_local[(size_t)f] = csc_indptr_local[(size_t)f + 1] - csc_indptr_local[(size_t)f];
        }
        std::vector<int64_t> df_global((size_t)F, 0);
        MPI_Allreduce(df_local.data(), df_global.data(), (int)F, MPI_LONG_LONG, MPI_SUM, comm);

        // ---- Candidate-generation feature mask (DF cap) ----
        if (df_cap_frac < 0.0) df_cap_frac = 0.0;
        if (df_cap_frac > 1.0) df_cap_frac = 1.0;

        std::vector<uint8_t> use_feat((size_t)F, 1u);

        int64_t df_cap = (int64_t)std::floor(df_cap_frac * (double)std::max<int64_t>(1, N_global));
        if (df_cap < 1) df_cap = 1;

        if (df_cap_frac < 1.0) {
            for (int64_t f = 0; f < F; ++f) {
                if (df_global[(size_t)f] > df_cap) use_feat[(size_t)f] = 0u;
            }
        }

        int64_t used = 0;
        for (auto u : use_feat) used += (u != 0);

        // Fallback: never allow "cap removes all features"
        if (used == 0) {
            std::fill(use_feat.begin(), use_feat.end(), (uint8_t)1);
            used = F;
            if (rank == 0 && verbose) {
                std::cout << "[knn] WARNING: df_cap removed all features; falling back to using all features.\n";
            }
        }

        if (rank == 0 && verbose) {
            std::cout << "[knn] N_global=" << N_global
                << " F=" << F
                << " df_cap_frac=" << df_cap_frac
                << " df_cap=" << df_cap
                << " used_features=" << used << "/" << F
                << " local_rows(rank0)=" << local_rows
                << "\n";
        }

        // Determine global max #blocks so all ranks participate in ring comm even if some have 0 rows
        const int64_t total_blocks_local = (block_size > 0) ? ((local_rows + block_size - 1) / block_size) : 0;
        int64_t total_blocks_max = 0;
        MPI_Allreduce((void*)&total_blocks_local, &total_blocks_max, 1, MPI_LONG_LONG, MPI_MAX, comm);

        // Ring neighbors
        const int right = (rank + 1) % world;
        const int left = (rank - 1 + world) % world;

        // Current peer buffers (start with local), and rotate during ring. After 'world' steps it returns to local.
        std::vector<int64_t> cur_indptr = csc_indptr_local;
        std::vector<int32_t> cur_indices = csc_indices_local;
        std::vector<int32_t> cur_deg = deg_local;
        int64_t cur_rows = local_rows;
        int64_t cur_row0 = row0_global;

        // Working buffers
        std::vector<int32_t> counts;
        std::vector<int32_t> touched;
        touched.reserve(4096);

        auto t0 = MPI_Wtime();

        // Main loop over blocks (GLOBAL block count)
        for (int64_t block_id = 0; block_id < total_blocks_max; ++block_id) {
            const int64_t block0 = block_id * (int64_t)block_size;
            const int64_t block1 = std::min(local_rows, block0 + (int64_t)block_size);
            const int64_t b = (block1 > block0) ? (block1 - block0) : 0;

            std::vector<TopK> topk;
            topk.reserve((size_t)b);
            for (int64_t t = 0; t < b; ++t) topk.emplace_back(k);

            // Ring across peers
            for (int step = 0; step < world; ++step) {

                // ensure counts sized (avoid O(cur_rows) clearing if size unchanged)
                if ((int64_t)counts.size() != cur_rows) {
                    counts.assign((size_t)cur_rows, 0);
                }

                // Compute only if this rank has rows in this block
                if (b > 0) {
                    for (int64_t off = 0; off < b; ++off) {
                        const int64_t i_local = block0 + off;
                        if (row_is_zero[(size_t)i_local]) continue;

                        const int64_t rs = B.indptr[(size_t)i_local];
                        const int64_t re = B.indptr[(size_t)i_local + 1];
                        const int32_t deg_i = (int32_t)(re - rs);
                        if (deg_i == 0) continue;

                        touched.clear();

                        // accumulate intersections
                        for (int64_t p = rs; p < re; ++p) {
                            const int32_t f = B.indices[(size_t)p];
                            if (!use_feat[(size_t)f]) continue;

                            const int64_t ps = cur_indptr[(size_t)f];
                            const int64_t pe = cur_indptr[(size_t)f + 1];

                            for (int64_t q = ps; q < pe; ++q) {
                                const int32_t jloc = cur_indices[(size_t)q];
                                // skip self when comparing with our own shard
                                if (cur_row0 == row0_global && jloc == i_local) continue;

                                if (counts[(size_t)jloc] == 0) touched.push_back(jloc);
                                ++counts[(size_t)jloc];
                            }
                        }

                        // compute similarities for touched candidates
                        for (int32_t jloc : touched) {
                            const int32_t inter = counts[(size_t)jloc];
                            const int32_t deg_j = (jloc >= 0 && (int64_t)jloc < cur_rows) ? cur_deg[(size_t)jloc] : 0;
                            const int32_t denom = deg_i + deg_j - inter;
                            float sim = (denom > 0) ? (float)inter / (float)denom : 0.0f;

                            if (sim > 0.0f) {
                                const int64_t j_global = cur_row0 + (int64_t)jloc;
                                topk[(size_t)off].consider(sim, j_global);
                            }
                            counts[(size_t)jloc] = 0;
                        }
                    }
                }

                // Rotate peer buffers (ring)
                if (world > 1) {
                    int64_t send_rows = cur_rows;
                    int64_t send_row0 = cur_row0;
                    int64_t send_idx_len = (int64_t)cur_indices.size();

                    int64_t recv_rows = 0, recv_row0 = 0, recv_idx_len = 0;

                    ring_sendrecv_sizes(comm, right, left,
                        send_rows, send_row0, send_idx_len,
                        recv_rows, recv_row0, recv_idx_len);

                    std::vector<int64_t> recv_indptr((size_t)F + 1, 0);
                    std::vector<int32_t> recv_indices((size_t)recv_idx_len, 0);
                    std::vector<int32_t> recv_deg((size_t)recv_rows, 0);

                    ring_sendrecv_arrays(comm, right, left,
                        cur_indptr.data(), (int)(F + 1),
                        cur_indices.data(), send_idx_len,
                        cur_deg.data(), send_rows,
                        recv_indptr.data(), (int)(F + 1),
                        recv_indices.data(), recv_idx_len,
                        recv_deg.data(), recv_rows);

                    cur_rows = recv_rows;
                    cur_row0 = recv_row0;
                    cur_indptr.swap(recv_indptr);
                    cur_indices.swap(recv_indices);
                    cur_deg.swap(recv_deg);
                }
            } // ring steps

            // Finalize block into CSR (only for existing local rows)
            if (b > 0) {
                for (int64_t off = 0; off < b; ++off) {
                    const int64_t base = (int64_t)G.indices.size();
                    topk[(size_t)off].finalize_sorted_desc(G.indices, G.weights);
                    G.indptr[(size_t)(block0 + off + 1)] = (int64_t)G.indices.size() - base;
                }
            }

            if (verbose && rank == 0 && b > 0) {
                double t1 = MPI_Wtime();
                std::cout << "[knn] block " << (block_id + 1) << "/" << total_blocks_max
                    << " done in " << (t1 - t0) << " s"
                    << " | k=" << k
                    << " | block_rows=" << b
                    << "\n";
                t0 = t1;
            }
        } // blocks

        // Prefix-sum indptr
        for (size_t r = 1; r < G.indptr.size(); ++r) {
            G.indptr[r] += G.indptr[r - 1];
        }

        // Optional: warn if graph is empty
        long long e_local = (long long)G.indices.size();
        long long e_global = 0;
        MPI_Allreduce(&e_local, &e_global, 1, MPI_LONG_LONG, MPI_SUM, comm);
        if (rank == 0 && verbose) {
            std::cout << "[knn] edges_global=" << e_global << " (directed)\n";
            if (e_global == 0 && df_cap_frac < 1.0) {
                std::cout << "[knn] WARNING: graph has 0 edges. Try df_cap_frac=1.0 (disable DF cap) or increase it.\n";
            }
        }

        return G;
    }

} // namespace rarecell
