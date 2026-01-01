#include "cluster/leiden.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <limits>

extern "C" {
#include <igraph.h>
#if __has_include(<igraph_version.h>)
#include <igraph_version.h>
#endif
}

namespace rarecell {

    static inline void mpi_check(int rc, const char* msg) {
        if (rc != MPI_SUCCESS) {
            throw std::runtime_error(std::string("MPI error: ") + msg);
        }
    }
    static inline void igraph_check(igraph_error_t rc, const char* where) {
        if (rc != IGRAPH_SUCCESS) {
            std::ostringstream oss;
            oss << "igraph error at " << where << ": " << igraph_strerror(rc);
            throw std::runtime_error(oss.str());
        }
    }

    

    struct EdgeUV {
        int64_t u;
        int64_t v;
        double  w;
    };
    static inline bool edge_less(const EdgeUV& a, const EdgeUV& b) {
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    }

    // Safer than sending structs (no padding issues)
    static void ring_sendrecv_sizes_unused_example() { /* not used here */ }

    std::vector<int32_t> leiden_cluster_mpi(const KNNGraphLocal& G_local,
        MPI_Comm comm,
        const LeidenParams& p)
    {
        int rank = 0, nprocs = 1;
        mpi_check(MPI_Comm_rank(comm, &rank), "Comm_rank");
        mpi_check(MPI_Comm_size(comm, &nprocs), "Comm_size");

        const int64_t n = G_local.n_nodes;
        const int64_t row0 = G_local.row0;
        const int64_t row1 = G_local.row1;
        const int64_t local_rows = row1 - row0;

        if (n <= 0) return {};
        if ((int64_t)G_local.indptr.size() != local_rows + 1) {
            throw std::runtime_error("Leiden: invalid KNNGraphLocal indptr size.");
        }
        if (G_local.indices.size() != G_local.weights.size()) {
            throw std::runtime_error("Leiden: indices/weights size mismatch.");
        }

        // ---------------------------
        // 1) Build LOCAL edge list
        // ---------------------------
        std::vector<int64_t> u_local;
        std::vector<int64_t> v_local;
        std::vector<double>  w_local;

        const int64_t nnz_local = (int64_t)G_local.weights.size();
        const int64_t reserve_guess = p.assume_symmetric ? (nnz_local / 2 + 1) : (nnz_local + 1);
        u_local.reserve((size_t)reserve_guess);
        v_local.reserve((size_t)reserve_guess);
        w_local.reserve((size_t)reserve_guess);

        const auto& indptr = G_local.indptr;
        const auto& idx = G_local.indices;
        const auto& wts_f = G_local.weights;

        for (int64_t r = 0; r < local_rows; ++r) {
            const int64_t i = row0 + r;
            const int64_t s = indptr[(size_t)r];
            const int64_t e = indptr[(size_t)r + 1];

            for (int64_t t = s; t < e; ++t) {
                const int64_t j = idx[(size_t)t];
                if (j < 0 || j >= n) continue;
                if (j == i) continue;

                double w = (double)wts_f[(size_t)t];

                // distance -> similarity if requested
                if (p.is_distance) {
                    w = 1.0 - w;
                    if (w < 0.0) w = 0.0;
                    if (w > 1.0) w = 1.0;
                }

                if (w <= 0.0) continue;

                if (p.assume_symmetric) {
                    // upper triangle only
                    if (j <= i) continue;
                    u_local.push_back(i);
                    v_local.push_back(j);
                    w_local.push_back(w);
                }
                else {
                    // map directed edge to undirected key (min,max)
                    const int64_t a = (i < j) ? i : j;
                    const int64_t b = (i < j) ? j : i;
                    u_local.push_back(a);
                    v_local.push_back(b);
                    w_local.push_back(w);
                }
            }
        }

        if (u_local.size() != v_local.size() || u_local.size() != w_local.size()) {
            throw std::runtime_error("Leiden: local edge vectors size mismatch.");
        }

        // MS-MPI uses int counts in Gatherv
        if (u_local.size() > (size_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error("Leiden: too many local edges for MPI_Gatherv (int counts). Chunking needed.");
        }
        const int m_local = (int)u_local.size();

        // ---------------------------
        // 2) Gather edge lists to rank0
        // ---------------------------
        std::vector<int> counts, displs;
        int m_total = 0;

        if (rank == 0) {
            counts.resize(nprocs);
            displs.resize(nprocs);
        }

        mpi_check(MPI_Gather(&m_local, 1, MPI_INT,
            rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
            0, comm),
            "Gather(m_local)");

        std::vector<int64_t> u_all, v_all;
        std::vector<double>  w_all;

        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < nprocs; ++r) displs[r] = displs[r - 1] + counts[r - 1];
            m_total = displs[nprocs - 1] + counts[nprocs - 1];

            u_all.resize((size_t)m_total);
            v_all.resize((size_t)m_total);
            w_all.resize((size_t)m_total);
        }

        mpi_check(MPI_Gatherv(u_local.data(), m_local, MPI_LONG_LONG,
            rank == 0 ? u_all.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_LONG_LONG, 0, comm),
            "Gatherv(u)");

        mpi_check(MPI_Gatherv(v_local.data(), m_local, MPI_LONG_LONG,
            rank == 0 ? v_all.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_LONG_LONG, 0, comm),
            "Gatherv(v)");

        mpi_check(MPI_Gatherv(w_local.data(), m_local, MPI_DOUBLE,
            rank == 0 ? w_all.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_DOUBLE, 0, comm),
            "Gatherv(w)");

        // ---------------------------
        // 3) Rank0: dedup/merge if needed, run Leiden
        // ---------------------------
        std::vector<int32_t> labels_global;

        if (rank == 0) {
            if (p.verbose) {
                std::cout << "[rank 0] Leiden: gathered edges=" << m_total
                    << " assume_symmetric=" << (p.assume_symmetric ? "true" : "false")
                    << " force_symmetrize=" << (p.force_symmetrize ? "true" : "false")
                    << "\n";
            }

            std::vector<EdgeUV> edges;
            edges.reserve((size_t)m_total);

            for (int i = 0; i < m_total; ++i) {
                int64_t uu = u_all[(size_t)i];
                int64_t vv = v_all[(size_t)i];
                if (uu == vv) continue;
                if (uu < 0 || vv < 0 || uu >= n || vv >= n) continue;
                double ww = w_all[(size_t)i];
                if (!(ww > 0.0)) continue;
                edges.push_back({ uu, vv, ww });
            }

            if (!p.assume_symmetric) {
                // merge duplicates of undirected edges
                std::sort(edges.begin(), edges.end(), edge_less);

                size_t out = 0;
                for (size_t i = 0; i < edges.size(); ++i) {
                    if (out == 0 || edges[i].u != edges[out - 1].u || edges[i].v != edges[out - 1].v) {
                        edges[out++] = edges[i];
                    }
                    else {
                        if (p.force_symmetrize) {
                            edges[out - 1].w = std::max(edges[out - 1].w, edges[i].w);
                        }
                        else {
                            edges[out - 1].w += edges[i].w;
                        }
                    }
                }
                edges.resize(out);

                if (p.verbose) {
                    std::cout << "[rank 0] Leiden: after merge edges=" << edges.size() << "\n";
                }
            }

            const igraph_int_t n_vertices = (igraph_int_t)n;
            const igraph_int_t m_edges = (igraph_int_t)edges.size();

            labels_global.assign((size_t)n, 0);

            if (m_edges > 0) {
                // Build edge vector
                igraph_vector_int_t edge_vec;
                igraph_vector_int_init(&edge_vec, 2 * m_edges);

                igraph_vector_t w_vec;
                igraph_vector_init(&w_vec, m_edges);

                // Strength as node weights (weighted degree)
                std::vector<double> strength((size_t)n, 0.0);
                double sum_w = 0.0;

                for (igraph_int_t e = 0; e < m_edges; ++e) {
                    const int64_t uu = edges[(size_t)e].u;
                    const int64_t vv = edges[(size_t)e].v;
                    const double  ww = edges[(size_t)e].w;

                    VECTOR(edge_vec)[2 * e + 0] = (igraph_int_t)uu;
                    VECTOR(edge_vec)[2 * e + 1] = (igraph_int_t)vv;
                    VECTOR(w_vec)[e] = (igraph_real_t)ww;

                    strength[(size_t)uu] += ww;
                    strength[(size_t)vv] += ww;
                    sum_w += ww;
                }
                if (!(sum_w > 0.0)) sum_w = 1.0;

                igraph_t g;
                igraph_check(igraph_create(&g, &edge_vec, n_vertices, /*directed=*/0), "igraph_create");

                // Node weights vector
                igraph_vector_t node_w;
                igraph_vector_init(&node_w, n_vertices);
                for (igraph_int_t i = 0; i < n_vertices; ++i) {
                    VECTOR(node_w)[i] = (igraph_real_t)strength[(size_t)i];
                }

                // Membership init (each node alone)
                igraph_vector_int_t membership;
                igraph_vector_int_init(&membership, n_vertices);
                for (igraph_int_t i = 0; i < n_vertices; ++i) VECTOR(membership)[i] = i;

                igraph_int_t nb_clusters = 0;
                igraph_real_t quality = 0.0;

                igraph_rng_seed(igraph_rng_default(), (igraph_uint_t)p.seed);

                double gamma_scaled = p.resolution / (2.0 * sum_w);

                // NOTE:
                // igraph's Leiden uses "resolution" on the RBConfiguration-like objective with node_weights.
                // If you find results too coarse/fine vs Python leidenalg, adjust p.resolution.
                // (We do NOT do extra gamma scaling here to keep behavior straightforward.)
#if defined(IGRAPH_VERSION_MAJOR) && (IGRAPH_VERSION_MAJOR >= 1)
                igraph_check(
                    igraph_community_leiden(
                        &g,
                        &w_vec,
                        &node_w,
                        /*vertex_in_weights=*/nullptr,
                        (igraph_real_t)gamma_scaled,
                        (igraph_real_t)p.beta,
                        /*start=*/0,
                        (igraph_int_t)p.n_iterations,
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
                        &w_vec,
                        &node_w,
                        (igraph_real_t)p.resolution,
                        (igraph_real_t)p.beta,
                        /*start=*/0,
                        (igraph_integer_t)p.n_iterations,
                        &membership,
                        &nb_clusters,
                        &quality
                    ),
                    "igraph_community_leiden(0.10.x)"
                );
#endif

                if (p.verbose) {
                    std::cout << "[rank 0] Leiden done: clusters=" << (long long)nb_clusters
                        << " quality=" << (double)quality
                        << " resolution=" << p.resolution
                        << "\n";
                }

                for (igraph_int_t i = 0; i < n_vertices; ++i) {
                    labels_global[(size_t)i] = (int32_t)VECTOR(membership)[i];
                }

                igraph_vector_int_destroy(&membership);
                igraph_vector_destroy(&node_w);
                igraph_destroy(&g);
                igraph_vector_destroy(&w_vec);
                igraph_vector_int_destroy(&edge_vec);
            }
        }

        // ---------------------------
        // 4) Scatter labels back (each rank gets its row slice)
        // ---------------------------
        const int local_rows_i = (int)local_rows;
        std::vector<int> row_counts, row_displs;

        if (rank == 0) { row_counts.resize(nprocs); row_displs.resize(nprocs); }

        mpi_check(MPI_Gather(&local_rows_i, 1, MPI_INT,
            rank == 0 ? row_counts.data() : nullptr, 1, MPI_INT,
            0, comm),
            "Gather(local_rows)");

        if (rank == 0) {
            row_displs[0] = 0;
            for (int r = 1; r < nprocs; ++r) row_displs[r] = row_displs[r - 1] + row_counts[r - 1];
            const int sum_rows = row_displs[nprocs - 1] + row_counts[nprocs - 1];
            if (sum_rows != (int)n) {
                std::ostringstream oss;
                oss << "Leiden scatter mismatch: sum(local_rows)=" << sum_rows << " but n=" << n;
                throw std::runtime_error(oss.str());
            }
        }

        std::vector<int32_t> labels_local((size_t)local_rows);

        mpi_check(MPI_Scatterv(rank == 0 ? labels_global.data() : nullptr,
            rank == 0 ? row_counts.data() : nullptr,
            rank == 0 ? row_displs.data() : nullptr,
            MPI_INT,
            labels_local.data(),
            local_rows_i,
            MPI_INT,
            0,
            comm),
            "Scatterv(labels)");

        return labels_local;
    }

} // namespace rarecell
