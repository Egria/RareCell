#include "graph/knn.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace rarecell {

    // Convert CSR (rows=cells, cols=features, binary) to CSC (cols=features)
    static void csr_to_csc_binary(const CSRMatrixU8& csr,
        std::vector<int64_t>& csc_indptr, // size F+1
        std::vector<int32_t>& csc_indices // length nnz
    ) {
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
        for (int64_t f = 0; f < F; ++f) csc_indptr[(size_t)f + 1] += csc_indptr[(size_t)f];

        const int64_t nnz = csc_indptr[(size_t)F];
        csc_indices.assign((size_t)nnz, 0);

        // Copy with next pointers
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

    // Ring send/recv helpers (use 'comm' explicitly)
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

    static void ring_sendrecv_arrays(MPI_Comm comm,
        int right, int left,
        const int64_t* send_indptr, int indptr_len,
        const int32_t* send_indices, int64_t indices_len,
        const int32_t* send_deg, int64_t rows_len,
        int64_t* recv_indptr, int recv_indptr_len,
        int32_t* recv_indices, int64_t recv_indices_len,
        int32_t* recv_deg, int64_t recv_rows_len)
    {
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
                heap.back() = { s,j };
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
        void finalize_sorted_desc(std::vector<int64_t>& out_idx, std::vector<float>& out_w) {
            std::sort_heap(heap.begin(), heap.end(), cmp); // ascending
            std::reverse(heap.begin(), heap.end());        // descending
            for (auto& p : heap) { out_idx.push_back(p.second); out_w.push_back(p.first); }
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
        const int64_t n_global = B.n_rows;

        KNNGraphLocal G;
        G.n_nodes = n_global;
        G.row0 = B.row0;
        G.row1 = B.row1;
        G.indptr.assign((size_t)local_rows + 1, 0);

        if (F <= 0 || n_global <= 0 || local_rows <= 0) {
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
        for (int64_t r = 0; r < local_rows; ++r) if (deg_local[(size_t)r] == 0) row_is_zero[(size_t)r] = 1u;

        // Global feature DF (per column nnz) to optionally cap ultra-common features
        // 1) local per-feature nnz:
        std::vector<int64_t> df_local((size_t)F, 0);
        for (int64_t f = 0; f < F; ++f) {
            df_local[(size_t)f] = csc_indptr_local[(size_t)f + 1] - csc_indptr_local[(size_t)f];
        }
        // 2) allreduce to global
        std::vector<int64_t> df_global((size_t)F, 0);
        MPI_Allreduce(df_local.data(), df_global.data(), (int)F, MPI_LONG_LONG, MPI_SUM, comm);

        // Pick feature mask for candidate generation
        const int64_t df_cap = (int64_t)std::floor(df_cap_frac * (double)std::max<int64_t>(1, n_global));
        std::vector<uint8_t> use_feat((size_t)F, 1u);
        if (df_cap_frac < 1.0) {
            for (int64_t f = 0; f < F; ++f) {
                if (df_global[(size_t)f] > df_cap) use_feat[(size_t)f] = 0u;
            }
        }

        // Output graph buffers (will append)
        std::vector<int64_t> out_indices;
        std::vector<float>   out_weights;
        out_indices.reserve((size_t)local_rows * std::min<int64_t>(k, 8));
        out_weights.reserve((size_t)local_rows * std::min<int64_t>(k, 8));

        // Ring allgather of CSC+degrees; process per block of local rows
        const int right = (rank + 1) % world;
        const int left = (rank - 1 + world) % world;

        // Current "peer" buffers start as our local data
        std::vector<int64_t> peer_indptr = csc_indptr_local;
        std::vector<int32_t> peer_indices = csc_indices_local;
        std::vector<int32_t> peer_deg = deg_local;
        int64_t peer_rows = local_rows;
        int64_t peer_row0 = B.row0;

        // All ranks must agree on F (panel size)
        int64_t F_check = F, F_min = 0;
        MPI_Allreduce(&F_check, &F_min, 1, MPI_LONG_LONG, MPI_MIN, comm);
        if (F_min != F) throw std::runtime_error("Panel features mismatch across ranks.");

        auto t0 = MPI_Wtime();
        const int64_t total_blocks = (local_rows + block_size - 1) / block_size;

        // Working buffers for counts per peer row (reused)
        std::vector<int32_t> counts;          // size = peer_rows
        std::vector<int32_t> touched;         // compact list of touched peer rows

        for (int64_t block_id = 0; block_id < total_blocks; ++block_id) {
            const int64_t block0 = block_id * (int64_t)block_size;
            const int64_t block1 = std::min(local_rows, block0 + (int64_t)block_size);
            const int64_t b = block1 - block0;

            // per-row top-k heap for this block
            std::vector<TopK> topk; topk.reserve((size_t)b);
            for (int64_t t = 0; t < b; ++t) topk.emplace_back(k);

            // Reset peer views for each block
            int64_t cur_rows = peer_rows;
            int64_t cur_row0 = peer_row0;
            std::vector<int64_t> cur_indptr = peer_indptr;
            std::vector<int32_t> cur_indices = peer_indices;
            std::vector<int32_t> cur_deg = peer_deg;

            for (int step = 0; step < world; ++step) {
                // Ensure work buffers sized to peer
                counts.assign((size_t)cur_rows, 0);
                touched.clear(); touched.reserve(4096);

                // For each local row in the block, accumulate intersections with this peer
                for (int64_t off = 0; off < b; ++off) {
                    const int64_t i_local = block0 + off;
                    if (row_is_zero[(size_t)i_local]) continue; // no work
                    const int64_t i_global = B.row0 + i_local;

                    const int64_t rs = B.indptr[(size_t)i_local];
                    const int64_t re = B.indptr[(size_t)i_local + 1];
                    const int32_t deg_i = (int32_t)(re - rs);
                    if (deg_i == 0) continue;

                    // Accumulate intersections through posting lists (skip ultra-common features)
                    for (int64_t p = rs; p < re; ++p) {
                        const int32_t f = B.indices[(size_t)p];
                        if (!use_feat[(size_t)f]) continue; // candidate cap
                        const int64_t ps = cur_indptr[(size_t)f];
                        const int64_t pe = cur_indptr[(size_t)f + 1];
                        for (int64_t q = ps; q < pe; ++q) {
                            const int32_t jloc = cur_indices[(size_t)q];
                            // skip self if this is our own shard
                            if (cur_row0 == B.row0 && jloc == i_local) continue;
                            if (counts[(size_t)jloc] == 0) touched.push_back(jloc);
                            ++counts[(size_t)jloc];
                        }
                    }

                    // Evaluate similarities for touched candidates
                    for (int32_t jloc : touched) {
                        const int32_t inter = counts[(size_t)jloc];
                        const int32_t deg_j = cur_deg[(size_t)jloc];
                        const int32_t denom = deg_i + deg_j - inter;
                        float sim = (denom > 0) ? (float)inter / (float)denom : 0.0f;
                        if (sim > 0.0f) {
                            const int64_t j_global = cur_row0 + (int64_t)jloc;
                            topk[(size_t)off].consider(sim, j_global);
                        }
                        counts[(size_t)jloc] = 0; // reset
                    }
                    touched.clear();
                }

                // Rotate peer buffers (ring): send to right, recv from left
                if (world > 1) {
                    // sizes first
                    int64_t send_rows = cur_rows, send_row0 = cur_row0;
                    int64_t send_idx_len = (int64_t)cur_indices.size();
                    int64_t recv_rows = 0, recv_row0 = 0, recv_idx_len = 0;

                    ring_sendrecv_sizes(comm, right, left,
                        send_rows, send_row0, send_idx_len,
                        recv_rows, recv_row0, recv_idx_len);

                    // allocate recv
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

                    // swap in received as next peer
                    cur_rows = recv_rows;
                    cur_row0 = recv_row0;
                    cur_indptr.swap(recv_indptr);
                    cur_indices.swap(recv_indices);
                    cur_deg.swap(recv_deg);
                }
            } // end ring over peers

            // Finalize this block: append to output CSR
            for (int64_t off = 0; off < b; ++off) {
                const int64_t base = (int64_t)G.indices.size();
                topk[(size_t)off].finalize_sorted_desc(G.indices, G.weights);
                G.indptr[(size_t)(block0 + off + 1)] = (int64_t)G.indices.size() - base;
            }

            if (verbose && rank == 0) {
                double t1 = MPI_Wtime();
                std::cout << "[knn] block " << (block_id + 1) << "/" << total_blocks
                    << " done in " << (t1 - t0) << " s"
                    << " | df_cap_frac=" << df_cap_frac
                    << " | local_rows=" << local_rows
                    << " | k=" << k << "\n";
                t0 = t1;
            }
        } // blocks

        // Prefix-sum indptr
        for (size_t r = 1; r < G.indptr.size(); ++r) {
            G.indptr[r] += G.indptr[r - 1];
        }
        return G;
    }

} // namespace rarecell
