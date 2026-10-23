/*
  ChIMES Calculator
  Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
  Contributing Author: Stan Moore (2025)
*/

#include<array>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>
#include <numeric>
#include <stdexcept>
#include <climits>

using namespace std;

#include "memory_kokkos.h"

/* ---------------------------------------------------------------------- */

template<class DeviceType>
chimesFFKokkos<DeviceType>::chimesFFKokkos() : chimesFF()
{
 eflag = vflag = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
chimesFFKokkos<DeviceType>::~chimesFFKokkos()
{

}

/* ---------------------------------------------------------------------- */


static inline void check_uniform_grid_1d_host(
    const std::vector<double> &x,
    double &x0,
    double &dx,
    double &invdx,
    const char *label,
    int idx_label
)
{
  if ((int)x.size() < 2) {
    std::cout << "ERROR: " << label << " table " << idx_label
              << " has fewer than 2 grid points" << std::endl;
    exit(1);
  }

  x0 = x[0];
  dx = x[1] - x[0];

  if (dx <= 0.0) {
    std::cout << "ERROR: " << label << " table " << idx_label
              << " has non-positive spacing" << std::endl;
    exit(1);
  }

  const double tol = 1.0e-10 * std::max(1.0, std::abs(dx));

  for (int i = 1; i < (int)x.size() - 1; i++) {
    double dxi = x[i+1] - x[i];
    if (std::abs(dxi - dx) > tol) {
      std::cout << "ERROR: " << label << " table " << idx_label
                << " is not uniform at i=" << i
                << " dx_ref=" << dx
                << " dx_i=" << dxi << std::endl;
      exit(1);
    }
  }

  invdx = 1.0 / dx;
}

static inline int infer_cube_n_host(int nrows, const char *label, int idx_label)
{
  int n = (int)std::round(std::cbrt((double)nrows));
  if (n * n * n != nrows) {
    std::cout << "ERROR: " << label << " table " << idx_label
              << " row count " << nrows
              << " is not a perfect cube" << std::endl;
    exit(1);
  }
  return n;
}

static inline int infer_svd3_grid_n_host(int nrows, int rank, const char *label, int idx_label)
{
  if (rank <= 0) {
    std::cout << "ERROR: " << label << " table " << idx_label
              << " has invalid rank " << rank << std::endl;
    exit(1);
  }
  int n = (int)std::round(std::cbrt((double)nrows));
  if (n * n * n != nrows) {
    std::cout << "ERROR: " << label << " table " << idx_label
              << " row count " << nrows
              << " is not a perfect cube" << std::endl;
    exit(1);
  }
  return n;
}

static inline int infer_svd4_left_n_host(int nrows, const char *label, int idx_label)
{
  int n = (int)std::round(std::sqrt(std::sqrt((double)nrows)));
  if (n * n * n * n != nrows) {
    std::cout << "ERROR: " << label << " left table " << idx_label
              << " row count " << nrows
              << " is not a perfect 4th power" << std::endl;
    exit(1);
  }
  return n;
}

static inline int infer_svd4_right_n_host(int nrows, const char *label, int idx_label)
{
  int n = (int)std::round(std::sqrt((double)nrows));
  if (n * n != nrows) {
    std::cout << "ERROR: " << label << " right table " << idx_label
              << " row count " << nrows
              << " is not a perfect square" << std::endl;
    exit(1);
  }
  return n;
}

template<class DeviceType>
void chimesFFKokkos<DeviceType>::read_parameters(string paramfile)
{
  int size, max_j, max_k;

  chimesFF::read_parameters(paramfile);
#ifdef TABULATION
  // ------------------------------------------------------------------
  // 4B CP 1D tabulation
  //
  // Slot tables are packed back to back, interleaved as on the host:
  //   vd[off + row*2*R + q]     = value      of rank term q at row
  //   vd[off + row*2*R + R + q] = derivative of rank term q at row
  // The kernels keep no per-rank arrays, so the rank is not limited.
  // ------------------------------------------------------------------
  if (tabulate_4B_cp) {
    tabulate_4B_cp_kk = true;

    const int nquads = (int)tab_4b_cp_rank.size();

    std::vector<int> h_rank(nquads, 0);
    std::vector<int> h_map(6 * nquads, 0);
    std::vector<int> h_ngrid(6 * nquads, 0);
    std::vector<int> h_offset(6 * nquads, 0);
    std::vector<double> h_r0(6 * nquads, 0.0);
    std::vector<double> h_dr(6 * nquads, 0.0);
    std::vector<double> h_invdr(6 * nquads, 0.0);

    size_t total = 0;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_cp_rank[q];
      h_rank[q] = R;

      if (R <= 0)
        continue;

      for (int c = 0; c < 6; c++) {
        h_map[6*q + c] =
          ((int)tab_4b_cp_canon_to_param[q].size() == 6)
          ? tab_4b_cp_canon_to_param[q][c] : c;

        const chimesFF::CP1DTable &tab = tab_4b_cp_slot[q][c];

        if (tab.rank != R || tab.vd.size() != (size_t)tab.ngrid * (size_t)(2 * R)) {
          std::cout << "ERROR: CP rank or table size mismatch for quad "
                    << q << " slot " << c << std::endl;
          exit(1);
        }

        h_ngrid[6*q + c] = tab.ngrid;
        h_offset[6*q + c] = (int)total;
        h_r0[6*q + c] = tab.r0;
        h_dr[6*q + c] = tab.dr;
        h_invdr[6*q + c] = tab.invdr;

        total += tab.vd.size();
      }
    }

    if (total > (size_t)INT_MAX) {
      std::cout << "ERROR: 4B CP tables too large (" << total << " entries)" << std::endl;
      exit(1);
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_rank,
                                     "chimesFF:tab4b_cp_rank", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_canon_to_param,
                                     "chimesFF:tab4b_cp_canon_to_param", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_ngrid,
                                     "chimesFF:tab4b_cp_ngrid", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_offset,
                                     "chimesFF:tab4b_cp_offset", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_r0,
                                     "chimesFF:tab4b_cp_r0", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_dr,
                                     "chimesFF:tab4b_cp_dr", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_invdr,
                                     "chimesFF:tab4b_cp_invdr", nquads, 6);

    d_tab4b_cp_vd = t_cp_tab_1d("chimesFF:tab4b_cp_vd", total);

    auto h_rank_v = Kokkos::create_mirror_view(d_tab4b_cp_rank);
    auto h_map_v = Kokkos::create_mirror_view(d_tab4b_cp_canon_to_param);
    auto h_ngrid_v = Kokkos::create_mirror_view(d_tab4b_cp_ngrid);
    auto h_offset_v = Kokkos::create_mirror_view(d_tab4b_cp_offset);
    auto h_r0_v = Kokkos::create_mirror_view(d_tab4b_cp_r0);
    auto h_dr_v = Kokkos::create_mirror_view(d_tab4b_cp_dr);
    auto h_invdr_v = Kokkos::create_mirror_view(d_tab4b_cp_invdr);
    auto h_vd_v = Kokkos::create_mirror_view(d_tab4b_cp_vd);

    for (int q = 0; q < nquads; q++) {
      h_rank_v[q] = h_rank[q];

      for (int c = 0; c < 6; c++) {
        h_map_v(q,c) = h_map[6*q + c];
        h_ngrid_v(q,c) = h_ngrid[6*q + c];
        h_offset_v(q,c) = h_offset[6*q + c];
        h_r0_v(q,c) = h_r0[6*q + c];
        h_dr_v(q,c) = h_dr[6*q + c];
        h_invdr_v(q,c) = h_invdr[6*q + c];
      }

      if (h_rank[q] <= 0) continue;

      for (int c = 0; c < 6; c++) {
        const std::vector<chimes_tab_real> &vd = tab_4b_cp_slot[q][c].vd;
        const size_t off = (size_t)h_offset[6*q + c];

        for (size_t i = 0; i < vd.size(); i++)
          h_vd_v[off + i] = vd[i];
      }
    }

    Kokkos::deep_copy(d_tab4b_cp_rank, h_rank_v);
    Kokkos::deep_copy(d_tab4b_cp_canon_to_param, h_map_v);
    Kokkos::deep_copy(d_tab4b_cp_ngrid, h_ngrid_v);
    Kokkos::deep_copy(d_tab4b_cp_offset, h_offset_v);
    Kokkos::deep_copy(d_tab4b_cp_r0, h_r0_v);
    Kokkos::deep_copy(d_tab4b_cp_dr, h_dr_v);
    Kokkos::deep_copy(d_tab4b_cp_invdr, h_invdr_v);
    Kokkos::deep_copy(d_tab4b_cp_vd, h_vd_v);

    c_tab4b_cp_rank = d_tab4b_cp_rank;
    c_tab4b_cp_canon_to_param = d_tab4b_cp_canon_to_param;
    c_tab4b_cp_ngrid = d_tab4b_cp_ngrid;
    c_tab4b_cp_offset = d_tab4b_cp_offset;
    c_tab4b_cp_r0 = d_tab4b_cp_r0;
    c_tab4b_cp_dr = d_tab4b_cp_dr;
    c_tab4b_cp_invdr = d_tab4b_cp_invdr;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 4B CP 1D tabulation, total entries = "
                << total << " (" << total * sizeof(chimes_tab_real) / 1048576.0
                << " MiB)" << std::endl;
  }
#endif
#ifdef TABULATION
  // ------------------------------------------------------------------
  // 4B CP direct/on-the-fly factors
  // ------------------------------------------------------------------
  if (tabulate_4B_cp_direct) {
    tabulate_4B_cp_direct_kk = true;

    const int nquads = (int)tab_4b_cp_direct_rank.size();

    std::vector<int> h_rank(nquads, 0);
    std::vector<int> h_map(6 * nquads, 0);
    std::vector<int> h_orders(6 * nquads, 0);
    std::vector<int> h_offset(6 * nquads, 0);

    int total = 0;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_cp_direct_rank[q];
      h_rank[q] = R;

      if (R <= 0)
        continue;

      for (int c = 0; c < 6; c++) {
        h_map[6*q + c] =
          ((int)tab_4b_cp_direct_canon_to_param[q].size() == 6)
          ? tab_4b_cp_direct_canon_to_param[q][c] : c;

        const int order = tab_4b_cp_direct_orders[q][c];

        if (order + 1 > MAX_4B_POLY) {
          std::cout << "ERROR: CP-direct order " << order
                    << " exceeds MAX_4B_POLY-1=" << MAX_4B_POLY-1
                    << std::endl;
          exit(1);
        }

        h_orders[6*q + c] = order;
        h_offset[6*q + c] = total;

        total += (order + 1) * R;
      }
    }

    std::vector<double> h_factor(total);

    for (int q = 0; q < nquads; q++) {
      const int R = h_rank[q];
      if (R <= 0) continue;

      for (int c = 0; c < 6; c++) {
        const int order = h_orders[6*q + c];
        const int off = h_offset[6*q + c];
        const std::vector<chimes_tab_real> &A = tab_4b_cp_direct_factor[q][c];

        const int nitems = (order + 1) * R;

        if ((int)A.size() != nitems) {
          std::cout << "ERROR: CP-direct factor size mismatch for quad "
                    << q << " slot " << c << std::endl;
          exit(1);
        }

        for (int i = 0; i < nitems; i++)
          h_factor[off + i] = A[i];
      }
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_direct_rank,
                                     "chimesFF:tab4b_cp_direct_rank", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_direct_canon_to_param,
                                     "chimesFF:tab4b_cp_direct_canon_to_param", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_direct_orders,
                                     "chimesFF:tab4b_cp_direct_orders", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_direct_offset,
                                     "chimesFF:tab4b_cp_direct_offset", nquads, 6);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_cp_direct_factor,
                                     "chimesFF:tab4b_cp_direct_factor", total);

    auto h_rank_v = Kokkos::create_mirror_view(d_tab4b_cp_direct_rank);
    auto h_map_v = Kokkos::create_mirror_view(d_tab4b_cp_direct_canon_to_param);
    auto h_orders_v = Kokkos::create_mirror_view(d_tab4b_cp_direct_orders);
    auto h_offset_v = Kokkos::create_mirror_view(d_tab4b_cp_direct_offset);
    auto h_factor_v = Kokkos::create_mirror_view(d_tab4b_cp_direct_factor);

    for (int q = 0; q < nquads; q++) {
      h_rank_v[q] = h_rank[q];

      for (int c = 0; c < 6; c++) {
        h_map_v(q,c) = h_map[6*q + c];
        h_orders_v(q,c) = h_orders[6*q + c];
        h_offset_v(q,c) = h_offset[6*q + c];
      }
    }

    for (int i = 0; i < total; i++)
      h_factor_v[i] = h_factor[i];

    Kokkos::deep_copy(d_tab4b_cp_direct_rank, h_rank_v);
    Kokkos::deep_copy(d_tab4b_cp_direct_canon_to_param, h_map_v);
    Kokkos::deep_copy(d_tab4b_cp_direct_orders, h_orders_v);
    Kokkos::deep_copy(d_tab4b_cp_direct_offset, h_offset_v);
    Kokkos::deep_copy(d_tab4b_cp_direct_factor, h_factor_v);

    c_tab4b_cp_direct_rank = d_tab4b_cp_direct_rank;
    c_tab4b_cp_direct_canon_to_param = d_tab4b_cp_direct_canon_to_param;
    c_tab4b_cp_direct_orders = d_tab4b_cp_direct_orders;
    c_tab4b_cp_direct_offset = d_tab4b_cp_direct_offset;
    c_tab4b_cp_direct_factor = d_tab4b_cp_direct_factor;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 4B CP-direct factors, total entries = "
                << total << std::endl;
  }
#endif

#ifdef TABULATION
  // ------------------------------------------------------------------
  // 3B CP 1D tabulation (same interleaved layout as the 4B CP tables; a
  // shared symmetric table is stored once and used by all three slots)
  // ------------------------------------------------------------------
  if (tabulate_3B_cp) {
    tabulate_3B_cp_kk = true;

    const int ntrips = (int)ncoeffs_3b.size();

    std::vector<int> h_rank(ntrips, 0);
    std::vector<int> h_map(3 * ntrips, 0);
    std::vector<int> h_ngrid(3 * ntrips, 0);
    std::vector<int> h_offset(3 * ntrips, 0);
    std::vector<double> h_r0(3 * ntrips, 0.0);
    std::vector<double> h_dr(3 * ntrips, 0.0);
    std::vector<double> h_invdr(3 * ntrips, 0.0);

    size_t total = 0;

    for (int t = 0; t < ntrips; t++) {
      const int R = ((int)tab_3b_cp.size() > t) ? tab_3b_cp[t].rank : 0;
      h_rank[t] = R;

      if (R <= 0) {
        if (ncoeffs_3b[t] > 0) {
          std::cout << "ERROR: no 3B CP table for triplet type " << t
                    << " (all triplet types must be tabulated when 3B CP TABLES is used)"
                    << std::endl;
          exit(1);
        }
        continue;
      }

      const chimesFF::CP3BTable &T = tab_3b_cp[t];

      for (int c = 0; c < 3; c++) {
        const chimesFF::CP1DTable &tab = T.slot[c];

        if (tab.rank != R || tab.vd.size() != (size_t)tab.ngrid * (size_t)(2 * R)) {
          std::cout << "ERROR: 3B CP rank or table size mismatch for triplet type "
                    << t << " slot " << c << std::endl;
          exit(1);
        }

        h_map[3*t + c] = T.canon_to_param[c];
        h_ngrid[3*t + c] = tab.ngrid;
        h_r0[3*t + c] = tab.r0;
        h_dr[3*t + c] = tab.dr;
        h_invdr[3*t + c] = tab.invdr;

        if (T.shared && c > 0) {
          h_offset[3*t + c] = h_offset[3*t];
        } else {
          h_offset[3*t + c] = (int)total;
          total += tab.vd.size();
        }
      }
    }

    if (total > (size_t)INT_MAX) {
      std::cout << "ERROR: 3B CP tables too large (" << total << " entries)" << std::endl;
      exit(1);
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_rank, "chimesFF:tab3b_cp_rank", ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_canon_to_param, "chimesFF:tab3b_cp_canon_to_param", ntrips, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_ngrid, "chimesFF:tab3b_cp_ngrid", ntrips, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_offset, "chimesFF:tab3b_cp_offset", ntrips, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_r0, "chimesFF:tab3b_cp_r0", ntrips, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_dr, "chimesFF:tab3b_cp_dr", ntrips, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_cp_invdr, "chimesFF:tab3b_cp_invdr", ntrips, 3);

    d_tab3b_cp_vd = t_cp_tab_1d("chimesFF:tab3b_cp_vd", total);

    auto h_rank_v = Kokkos::create_mirror_view(d_tab3b_cp_rank);
    auto h_map_v = Kokkos::create_mirror_view(d_tab3b_cp_canon_to_param);
    auto h_ngrid_v = Kokkos::create_mirror_view(d_tab3b_cp_ngrid);
    auto h_offset_v = Kokkos::create_mirror_view(d_tab3b_cp_offset);
    auto h_r0_v = Kokkos::create_mirror_view(d_tab3b_cp_r0);
    auto h_dr_v = Kokkos::create_mirror_view(d_tab3b_cp_dr);
    auto h_invdr_v = Kokkos::create_mirror_view(d_tab3b_cp_invdr);
    auto h_vd_v = Kokkos::create_mirror_view(d_tab3b_cp_vd);

    for (int t = 0; t < ntrips; t++) {
      h_rank_v[t] = h_rank[t];

      for (int c = 0; c < 3; c++) {
        h_map_v(t,c) = h_map[3*t + c];
        h_ngrid_v(t,c) = h_ngrid[3*t + c];
        h_offset_v(t,c) = h_offset[3*t + c];
        h_r0_v(t,c) = h_r0[3*t + c];
        h_dr_v(t,c) = h_dr[3*t + c];
        h_invdr_v(t,c) = h_invdr[3*t + c];
      }

      if (h_rank[t] <= 0) continue;

      const int nstore = tab_3b_cp[t].shared ? 1 : 3;

      for (int c = 0; c < nstore; c++) {
        const std::vector<chimes_tab_real> &vd = tab_3b_cp[t].slot[c].vd;
        const size_t off = (size_t)h_offset[3*t + c];

        for (size_t i = 0; i < vd.size(); i++)
          h_vd_v[off + i] = vd[i];
      }
    }

    Kokkos::deep_copy(d_tab3b_cp_rank, h_rank_v);
    Kokkos::deep_copy(d_tab3b_cp_canon_to_param, h_map_v);
    Kokkos::deep_copy(d_tab3b_cp_ngrid, h_ngrid_v);
    Kokkos::deep_copy(d_tab3b_cp_offset, h_offset_v);
    Kokkos::deep_copy(d_tab3b_cp_r0, h_r0_v);
    Kokkos::deep_copy(d_tab3b_cp_dr, h_dr_v);
    Kokkos::deep_copy(d_tab3b_cp_invdr, h_invdr_v);
    Kokkos::deep_copy(d_tab3b_cp_vd, h_vd_v);

    c_tab3b_cp_rank = d_tab3b_cp_rank;
    c_tab3b_cp_canon_to_param = d_tab3b_cp_canon_to_param;
    c_tab3b_cp_ngrid = d_tab3b_cp_ngrid;
    c_tab3b_cp_offset = d_tab3b_cp_offset;
    c_tab3b_cp_r0 = d_tab3b_cp_r0;
    c_tab3b_cp_dr = d_tab3b_cp_dr;
    c_tab3b_cp_invdr = d_tab3b_cp_invdr;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 3B CP 1D tabulation, total entries = "
                << total << " (" << total * sizeof(chimes_tab_real) / 1048576.0
                << " MiB)" << std::endl;
  }
#endif

#ifdef TABULATION
  tabulate_4B_svd3x3_corner_kk =
      tabulate_4B_svd3x3 &&
      tab_4b_svd3x3_left.size() > 0 &&
      tab_4b_svd3x3_left[0].corner_table;

  tabulate_4B_svd4x2_corner_kk =
      tabulate_4B_svd4x2 &&
      tab_4b_svd4x2_left.size() > 0 &&
      tab_4b_svd4x2_left[0].corner_table;
#endif
  #ifdef TABULATION
  // ------------------------------------------------------------------
  // 2B tabulation
  // ------------------------------------------------------------------
  if (tabulate_2B) {
    tabulate_2B_kk = true;

    const int npairs = (int)tab_r.size();

    if ((int)tab_e.size() != npairs || (int)tab_f.size() != npairs) {
      std::cout << "ERROR: 2B tabulation size mismatch" << std::endl;
      exit(1);
    }

    std::vector<int> h_n(npairs);
    std::vector<double> h_r0(npairs), h_dr(npairs), h_invdr(npairs);
    std::vector<int> h_offset(npairs);

    int total_n = 0;
    for (int p = 0; p < npairs; p++) {
      if (tab_r[p].size() != tab_e[p].size() || tab_r[p].size() != tab_f[p].size()) {
        std::cout << "ERROR: 2B table length mismatch for pair " << p << std::endl;
        exit(1);
      }

      h_offset[p] = total_n;
      h_n[p] = (int)tab_r[p].size();

      check_uniform_grid_1d_host(tab_r[p], h_r0[p], h_dr[p], h_invdr[p], "2B", p);

      total_n += h_n[p];
    }

    std::vector<float> h_e_all(total_n);
    std::vector<float> h_f_all(total_n);

    for (int p = 0; p < npairs; p++) {
      const int off = h_offset[p];
      const int n = h_n[p];
      for (int i = 0; i < n; i++) {
        h_e_all[off + i] = (float)tab_e[p][i];
        h_f_all[off + i] = (float)tab_f[p][i];
      }
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab2b_n,      "chimesFF:tab2b_n",      npairs);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab2b_r0,     "chimesFF:tab2b_r0",     npairs);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab2b_dr,     "chimesFF:tab2b_dr",     npairs);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab2b_invdr,  "chimesFF:tab2b_invdr",  npairs);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab2b_offset, "chimesFF:tab2b_offset", npairs);

    d_tab2b_e = Kokkos::View<float*>("chimesFF:tab2b_e", total_n);
    d_tab2b_f = Kokkos::View<float*>("chimesFF:tab2b_f", total_n);

    auto h_tab2b_n      = Kokkos::create_mirror_view(d_tab2b_n);
    auto h_tab2b_r0     = Kokkos::create_mirror_view(d_tab2b_r0);
    auto h_tab2b_dr     = Kokkos::create_mirror_view(d_tab2b_dr);
    auto h_tab2b_invdr  = Kokkos::create_mirror_view(d_tab2b_invdr);
    auto h_tab2b_offset = Kokkos::create_mirror_view(d_tab2b_offset);
    auto h_tab2b_e      = Kokkos::create_mirror_view(d_tab2b_e);
    auto h_tab2b_f      = Kokkos::create_mirror_view(d_tab2b_f);

    for (int p = 0; p < npairs; p++) {
      h_tab2b_n[p] = h_n[p];
      h_tab2b_r0[p] = h_r0[p];
      h_tab2b_dr[p] = h_dr[p];
      h_tab2b_invdr[p] = h_invdr[p];
      h_tab2b_offset[p] = h_offset[p];
    }

    for (int i = 0; i < total_n; i++) {
      h_tab2b_e[i] = h_e_all[i];
      h_tab2b_f[i] = h_f_all[i];
    }

    Kokkos::deep_copy(d_tab2b_n,      h_tab2b_n);
    Kokkos::deep_copy(d_tab2b_r0,     h_tab2b_r0);
    Kokkos::deep_copy(d_tab2b_dr,     h_tab2b_dr);
    Kokkos::deep_copy(d_tab2b_invdr,  h_tab2b_invdr);
    Kokkos::deep_copy(d_tab2b_offset, h_tab2b_offset);
    Kokkos::deep_copy(d_tab2b_e,      h_tab2b_e);
    Kokkos::deep_copy(d_tab2b_f,      h_tab2b_f);

    c_tab2b_n      = d_tab2b_n;
    c_tab2b_r0     = d_tab2b_r0;
    c_tab2b_dr     = d_tab2b_dr;
    c_tab2b_invdr  = d_tab2b_invdr;
    c_tab2b_offset = d_tab2b_offset;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 2B tabulation, total points = "
                << total_n << std::endl;
  }
#endif

#ifdef TABULATION
  // ------------------------------------------------------------------
  // 3B tabulation
  // ------------------------------------------------------------------
  if (tabulate_3B) {
    tabulate_3B_kk = true;

    const int ntrips = (int)tab_e_3B.size();

    if ((int)tab_f_ij_3B.size() != ntrips ||
        (int)tab_f_ik_3B.size() != ntrips ||
        (int)tab_f_jk_3B.size() != ntrips) {
      std::cout << "ERROR: 3B tabulation size mismatch" << std::endl;
      exit(1);
    }

    std::vector<int> h_n(ntrips);
    std::vector<double> h_r0(ntrips), h_dr(ntrips), h_invdr(ntrips);
    std::vector<int> h_offset(ntrips);
    std::vector<int> h_canon_to_param(3 * ntrips, 0);

    int total_n = 0;

    for (int t = 0; t < ntrips; t++) {
      const int nrows = (int)tab_e_3B[t].size();

      if ((int)tab_f_ij_3B[t].size() != nrows ||
          (int)tab_f_ik_3B[t].size() != nrows ||
          (int)tab_f_jk_3B[t].size() != nrows ||
          (int)tab_rij_3B[t].size() != nrows ||
          (int)tab_rik_3B[t].size() != nrows ||
          (int)tab_rjk_3B[t].size() != nrows) {
        std::cout << "ERROR: 3B table size mismatch for trip " << t << std::endl;
        exit(1);
      }

      const int n = infer_cube_n_host(nrows, "3B", t);
      h_n[t] = n;
      h_offset[t] = total_n;

      // assume same spacing in all three dimensions
      // use first axis from tab_rjk progression
      double r0, dr, invdr;
      // Construct 1D axis from the first z-line
      std::vector<double> axis(n);
      for (int k = 0; k < n; k++) axis[k] = tab_rjk_3B[t][k];
      check_uniform_grid_1d_host(axis, r0, dr, invdr, "3B", t);

      h_r0[t] = r0;
      h_dr[t] = dr;
      h_invdr[t] = invdr;

      // canonical order by sorted pair-type strings
      std::vector<std::pair<std::string,int>> pairs(3);
      pairs[0] = {trip_params_pair_typs[t][0], 0};
      pairs[1] = {trip_params_pair_typs[t][1], 1};
      pairs[2] = {trip_params_pair_typs[t][2], 2};

      std::stable_sort(pairs.begin(), pairs.end(),
        [](const std::pair<std::string,int> &a, const std::pair<std::string,int> &b) {
          return a.first < b.first;
        });

      h_canon_to_param[3*t + 0] = pairs[0].second;
      h_canon_to_param[3*t + 1] = pairs[1].second;
      h_canon_to_param[3*t + 2] = pairs[2].second;

      total_n += nrows;
    }

    std::vector<float> h_e_all(total_n);
    std::vector<float> h_f0_all(total_n);
    std::vector<float> h_f1_all(total_n);
    std::vector<float> h_f2_all(total_n);

    for (int t = 0; t < ntrips; t++) {
      const int off = h_offset[t];
      const int nrows = (int)tab_e_3B[t].size();

      const int p0 = h_canon_to_param[3*t + 0];
      const int p1 = h_canon_to_param[3*t + 1];
      const int p2 = h_canon_to_param[3*t + 2];

      // select which CPU force table corresponds to canonical slot
      const std::vector<double> *srcF[3] = {nullptr,nullptr,nullptr};
      const std::vector<double> *allF[3] = {&tab_f_ij_3B[t], &tab_f_ik_3B[t], &tab_f_jk_3B[t]};

      srcF[0] = allF[p0];
      srcF[1] = allF[p1];
      srcF[2] = allF[p2];

      for (int i = 0; i < nrows; i++) {
        h_e_all [off + i] = (float)tab_e_3B[t][i];
        h_f0_all[off + i] = (float)(*srcF[0])[i];
        h_f1_all[off + i] = (float)(*srcF[1])[i];
        h_f2_all[off + i] = (float)(*srcF[2])[i];
      }
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_n,      "chimesFF:tab3b_n",      ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_r0,     "chimesFF:tab3b_r0",     ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_dr,     "chimesFF:tab3b_dr",     ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_invdr,  "chimesFF:tab3b_invdr",  ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_offset, "chimesFF:tab3b_offset", ntrips);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab3b_canon_to_param, "chimesFF:tab3b_canon_to_param", ntrips, 3);

    d_tab3b_e  = Kokkos::View<float*>("chimesFF:tab3b_e",  total_n);
    d_tab3b_f0 = Kokkos::View<float*>("chimesFF:tab3b_f0", total_n);
    d_tab3b_f1 = Kokkos::View<float*>("chimesFF:tab3b_f1", total_n);
    d_tab3b_f2 = Kokkos::View<float*>("chimesFF:tab3b_f2", total_n);

    auto h_tab3b_n      = Kokkos::create_mirror_view(d_tab3b_n);
    auto h_tab3b_r0     = Kokkos::create_mirror_view(d_tab3b_r0);
    auto h_tab3b_dr     = Kokkos::create_mirror_view(d_tab3b_dr);
    auto h_tab3b_invdr  = Kokkos::create_mirror_view(d_tab3b_invdr);
    auto h_tab3b_offset = Kokkos::create_mirror_view(d_tab3b_offset);
    auto h_tab3b_map    = Kokkos::create_mirror_view(d_tab3b_canon_to_param);
    auto h_tab3b_e      = Kokkos::create_mirror_view(d_tab3b_e);
    auto h_tab3b_f0     = Kokkos::create_mirror_view(d_tab3b_f0);
    auto h_tab3b_f1     = Kokkos::create_mirror_view(d_tab3b_f1);
    auto h_tab3b_f2     = Kokkos::create_mirror_view(d_tab3b_f2);

    for (int t = 0; t < ntrips; t++) {
      h_tab3b_n[t] = h_n[t];
      h_tab3b_r0[t] = h_r0[t];
      h_tab3b_dr[t] = h_dr[t];
      h_tab3b_invdr[t] = h_invdr[t];
      h_tab3b_offset[t] = h_offset[t];
      for (int j = 0; j < 3; j++)
        h_tab3b_map(t,j) = h_canon_to_param[3*t + j];
    }

    for (int i = 0; i < total_n; i++) {
      h_tab3b_e[i]  = h_e_all[i];
      h_tab3b_f0[i] = h_f0_all[i];
      h_tab3b_f1[i] = h_f1_all[i];
      h_tab3b_f2[i] = h_f2_all[i];
    }

    Kokkos::deep_copy(d_tab3b_n,      h_tab3b_n);
    Kokkos::deep_copy(d_tab3b_r0,     h_tab3b_r0);
    Kokkos::deep_copy(d_tab3b_dr,     h_tab3b_dr);
    Kokkos::deep_copy(d_tab3b_invdr,  h_tab3b_invdr);
    Kokkos::deep_copy(d_tab3b_offset, h_tab3b_offset);
    Kokkos::deep_copy(d_tab3b_canon_to_param, h_tab3b_map);
    Kokkos::deep_copy(d_tab3b_e,      h_tab3b_e);
    Kokkos::deep_copy(d_tab3b_f0,     h_tab3b_f0);
    Kokkos::deep_copy(d_tab3b_f1,     h_tab3b_f1);
    Kokkos::deep_copy(d_tab3b_f2,     h_tab3b_f2);

    c_tab3b_n      = d_tab3b_n;
    c_tab3b_r0     = d_tab3b_r0;
    c_tab3b_dr     = d_tab3b_dr;
    c_tab3b_invdr  = d_tab3b_invdr;
    c_tab3b_offset = d_tab3b_offset;
    c_tab3b_canon_to_param = d_tab3b_canon_to_param;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 3B tabulation, total points = "
                << total_n << std::endl;
  }
#endif

#ifdef TABULATION
  // ------------------------------------------------------------------
  // 4B SVD3x3 tabulation
  // ------------------------------------------------------------------
  if (tabulate_4B_svd3x3) {
    tabulate_4B_svd3x3_kk = true;

    const int nquads = (int)tab_4b_svd3x3_rank.size();

        bool any_svd3_corner = false;
    bool any_svd3_old = false;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_svd3x3_rank[q];

      // Skip unused/uninitialized quad entries.
      if (R <= 0)
        continue;

      const auto &L = tab_4b_svd3x3_left[q];
      const auto &Rt = tab_4b_svd3x3_right[q];

      if (L.corner_table != Rt.corner_table) {
        std::cout << "ERROR: SVD3x3 left/right corner table layout mismatch "
                  << "for quad " << q << std::endl;
        exit(1);
      }

      if (L.corner_table)
        any_svd3_corner = true;
      else
        any_svd3_old = true;
    }

    if (any_svd3_corner && any_svd3_old) {
      std::cout << "ERROR: Mixed old/corner SVD3x3 table layouts are not "
                << "currently supported in one run." << std::endl;
      exit(1);
    }

    tabulate_4B_svd3x3_corner_kk = any_svd3_corner;

    std::vector<int> h_rank(nquads), h_ngrid(nquads);
    std::vector<int> h_off_left(nquads), h_off_right(nquads);
    std::vector<int> h_canon_to_param(6 * nquads, 0);

    std::vector<double> h_r0_left(3 * nquads), h_dr_left(3 * nquads), h_invdr_left(3 * nquads);
    std::vector<double> h_r0_right(3 * nquads), h_dr_right(3 * nquads), h_invdr_right(3 * nquads);

    int total_left = 0;
    int total_right = 0;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_svd3x3_rank[q];
      h_rank[q] = R;

      const auto &L = tab_4b_svd3x3_left[q];
      const auto &Rtab = tab_4b_svd3x3_right[q];

      if (L.rank != R || Rtab.rank != R) {
        std::cout << "ERROR: SVD3x3 rank mismatch for quad " << q << std::endl;
        exit(1);
      }

      const int nL = L.ngrid;
      const int nR = Rtab.ngrid;

      if (nL != L.ngrid || nR != Rtab.ngrid || nL != nR) {
        std::cout << "ERROR: SVD3x3 grid mismatch for quad " << q << std::endl;
        exit(1);
      }

      h_ngrid[q] = nL;
      h_off_left[q] = total_left;
      h_off_right[q] = total_right;

      for (int d = 0; d < 3; d++) {
        h_r0_left [3*q + d] = L.r0[d];
        h_dr_left [3*q + d] = L.dr[d];
        h_invdr_left[3*q + d] = L.invdr[d];

        h_r0_right [3*q + d] = Rtab.r0[d];
        h_dr_right [3*q + d] = Rtab.dr[d];
        h_invdr_right[3*q + d] = Rtab.invdr[d];
      }

      for (int c = 0; c < 6; c++) {
        h_canon_to_param[6*q + c] =
          (tab_4b_svd3x3_canon_to_param[q].size() == 6)
          ? tab_4b_svd3x3_canon_to_param[q][c] : c;
      }

      total_left  += (int)L.val.size();
      total_right += (int)Rtab.val.size();
    }

    std::vector<float> h_left_val(total_left), h_left_d0(total_left), h_left_d1(total_left), h_left_d2(total_left);
    std::vector<float> h_right_val(total_right), h_right_d0(total_right), h_right_d1(total_right), h_right_d2(total_right);

    for (int q = 0; q < nquads; q++) {
      const int R = h_rank[q];
      const int offL = h_off_left[q];
      const int offR = h_off_right[q];

      const auto &L = tab_4b_svd3x3_left[q];
      const auto &Rt = tab_4b_svd3x3_right[q];

      for (int i = 0; i < (int)L.val.size(); i++) {
        h_left_val[offL + i] = L.val[i];
        h_left_d0 [offL + i] = L.d0[i];
        h_left_d1 [offL + i] = L.d1[i];
        h_left_d2 [offL + i] = L.d2[i];
      }

      for (int i = 0; i < (int)Rt.val.size(); i++) {
        h_right_val[offR + i] = Rt.val[i];
        h_right_d0 [offR + i] = Rt.d0[i];
        h_right_d1 [offR + i] = Rt.d1[i];
        h_right_d2 [offR + i] = Rt.d2[i];
      }
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_rank, "chimesFF:tab4b_svd3_rank", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_ngrid, "chimesFF:tab4b_svd3_ngrid", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_offset_left, "chimesFF:tab4b_svd3_offset_left", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_offset_right, "chimesFF:tab4b_svd3_offset_right", nquads);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_r0_left,    "chimesFF:tab4b_svd3_r0_left",    nquads, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_dr_left,    "chimesFF:tab4b_svd3_dr_left",    nquads, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_invdr_left, "chimesFF:tab4b_svd3_invdr_left", nquads, 3);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_r0_right,    "chimesFF:tab4b_svd3_r0_right",    nquads, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_dr_right,    "chimesFF:tab4b_svd3_dr_right",    nquads, 3);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_invdr_right, "chimesFF:tab4b_svd3_invdr_right", nquads, 3);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd3_canon_to_param, "chimesFF:tab4b_svd3_canon_to_param", nquads, 6);

    d_tab4b_svd3_left_val  = Kokkos::View<float*>("chimesFF:tab4b_svd3_left_val", total_left);
    d_tab4b_svd3_left_d0   = Kokkos::View<float*>("chimesFF:tab4b_svd3_left_d0", total_left);
    d_tab4b_svd3_left_d1   = Kokkos::View<float*>("chimesFF:tab4b_svd3_left_d1", total_left);
    d_tab4b_svd3_left_d2   = Kokkos::View<float*>("chimesFF:tab4b_svd3_left_d2", total_left);

    d_tab4b_svd3_right_val = Kokkos::View<float*>("chimesFF:tab4b_svd3_right_val", total_right);
    d_tab4b_svd3_right_d0  = Kokkos::View<float*>("chimesFF:tab4b_svd3_right_d0", total_right);
    d_tab4b_svd3_right_d1  = Kokkos::View<float*>("chimesFF:tab4b_svd3_right_d1", total_right);
    d_tab4b_svd3_right_d2  = Kokkos::View<float*>("chimesFF:tab4b_svd3_right_d2", total_right);

    auto h_rank_v  = Kokkos::create_mirror_view(d_tab4b_svd3_rank);
    auto h_ngrid_v = Kokkos::create_mirror_view(d_tab4b_svd3_ngrid);
    auto h_offL_v  = Kokkos::create_mirror_view(d_tab4b_svd3_offset_left);
    auto h_offR_v  = Kokkos::create_mirror_view(d_tab4b_svd3_offset_right);

    auto h_r0L_v   = Kokkos::create_mirror_view(d_tab4b_svd3_r0_left);
    auto h_drL_v   = Kokkos::create_mirror_view(d_tab4b_svd3_dr_left);
    auto h_ivL_v   = Kokkos::create_mirror_view(d_tab4b_svd3_invdr_left);

    auto h_r0R_v   = Kokkos::create_mirror_view(d_tab4b_svd3_r0_right);
    auto h_drR_v   = Kokkos::create_mirror_view(d_tab4b_svd3_dr_right);
    auto h_ivR_v   = Kokkos::create_mirror_view(d_tab4b_svd3_invdr_right);

    auto h_map_v   = Kokkos::create_mirror_view(d_tab4b_svd3_canon_to_param);

    auto h_Lv      = Kokkos::create_mirror_view(d_tab4b_svd3_left_val);
    auto h_Ld0     = Kokkos::create_mirror_view(d_tab4b_svd3_left_d0);
    auto h_Ld1     = Kokkos::create_mirror_view(d_tab4b_svd3_left_d1);
    auto h_Ld2     = Kokkos::create_mirror_view(d_tab4b_svd3_left_d2);

    auto h_Rv      = Kokkos::create_mirror_view(d_tab4b_svd3_right_val);
    auto h_Rd0     = Kokkos::create_mirror_view(d_tab4b_svd3_right_d0);
    auto h_Rd1     = Kokkos::create_mirror_view(d_tab4b_svd3_right_d1);
    auto h_Rd2     = Kokkos::create_mirror_view(d_tab4b_svd3_right_d2);

    for (int q = 0; q < nquads; q++) {
      h_rank_v[q] = h_rank[q];
      h_ngrid_v[q] = h_ngrid[q];
      h_offL_v[q] = h_off_left[q];
      h_offR_v[q] = h_off_right[q];
      for (int d = 0; d < 3; d++) {
        h_r0L_v(q,d) = h_r0_left[3*q + d];
        h_drL_v(q,d) = h_dr_left[3*q + d];
        h_ivL_v(q,d) = h_invdr_left[3*q + d];
        h_r0R_v(q,d) = h_r0_right[3*q + d];
        h_drR_v(q,d) = h_dr_right[3*q + d];
        h_ivR_v(q,d) = h_invdr_right[3*q + d];
      }
      for (int c = 0; c < 6; c++) h_map_v(q,c) = h_canon_to_param[6*q + c];
    }

    for (int i = 0; i < total_left; i++) {
      h_Lv[i]  = h_left_val[i];
      h_Ld0[i] = h_left_d0[i];
      h_Ld1[i] = h_left_d1[i];
      h_Ld2[i] = h_left_d2[i];
    }
    for (int i = 0; i < total_right; i++) {
      h_Rv[i]  = h_right_val[i];
      h_Rd0[i] = h_right_d0[i];
      h_Rd1[i] = h_right_d1[i];
      h_Rd2[i] = h_right_d2[i];
    }

    Kokkos::deep_copy(d_tab4b_svd3_rank, h_rank_v);
    Kokkos::deep_copy(d_tab4b_svd3_ngrid, h_ngrid_v);
    Kokkos::deep_copy(d_tab4b_svd3_offset_left, h_offL_v);
    Kokkos::deep_copy(d_tab4b_svd3_offset_right, h_offR_v);
    Kokkos::deep_copy(d_tab4b_svd3_r0_left, h_r0L_v);
    Kokkos::deep_copy(d_tab4b_svd3_dr_left, h_drL_v);
    Kokkos::deep_copy(d_tab4b_svd3_invdr_left, h_ivL_v);
    Kokkos::deep_copy(d_tab4b_svd3_r0_right, h_r0R_v);
    Kokkos::deep_copy(d_tab4b_svd3_dr_right, h_drR_v);
    Kokkos::deep_copy(d_tab4b_svd3_invdr_right, h_ivR_v);
    Kokkos::deep_copy(d_tab4b_svd3_canon_to_param, h_map_v);

    Kokkos::deep_copy(d_tab4b_svd3_left_val, h_Lv);
    Kokkos::deep_copy(d_tab4b_svd3_left_d0,  h_Ld0);
    Kokkos::deep_copy(d_tab4b_svd3_left_d1,  h_Ld1);
    Kokkos::deep_copy(d_tab4b_svd3_left_d2,  h_Ld2);

    Kokkos::deep_copy(d_tab4b_svd3_right_val, h_Rv);
    Kokkos::deep_copy(d_tab4b_svd3_right_d0,  h_Rd0);
    Kokkos::deep_copy(d_tab4b_svd3_right_d1,  h_Rd1);
    Kokkos::deep_copy(d_tab4b_svd3_right_d2,  h_Rd2);

    c_tab4b_svd3_rank = d_tab4b_svd3_rank;
    c_tab4b_svd3_ngrid = d_tab4b_svd3_ngrid;
    c_tab4b_svd3_offset_left = d_tab4b_svd3_offset_left;
    c_tab4b_svd3_offset_right = d_tab4b_svd3_offset_right;
    c_tab4b_svd3_r0_left = d_tab4b_svd3_r0_left;
    c_tab4b_svd3_dr_left = d_tab4b_svd3_dr_left;
    c_tab4b_svd3_invdr_left = d_tab4b_svd3_invdr_left;
    c_tab4b_svd3_r0_right = d_tab4b_svd3_r0_right;
    c_tab4b_svd3_dr_right = d_tab4b_svd3_dr_right;
    c_tab4b_svd3_invdr_right = d_tab4b_svd3_invdr_right;
    c_tab4b_svd3_canon_to_param = d_tab4b_svd3_canon_to_param;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 4B SVD3x3 tabulation" << std::endl;
  }
#endif

#ifdef TABULATION
  // ------------------------------------------------------------------
  // 4B SVD4x2 tabulation
  // ------------------------------------------------------------------
  if (tabulate_4B_svd4x2) {
    tabulate_4B_svd4x2_kk = true;

    const int nquads = (int)tab_4b_svd4x2_rank.size();

        bool any_svd42_corner = false;
    bool any_svd42_old = false;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_svd4x2_rank[q];

      // Skip unused/uninitialized quad entries.
      if (R <= 0)
        continue;

      const auto &L = tab_4b_svd4x2_left[q];
      const auto &Rt = tab_4b_svd4x2_right[q];

      if (L.corner_table != Rt.corner_table) {
        std::cout << "ERROR: SVD4x2 left/right corner table layout mismatch "
                  << "for quad " << q << std::endl;
        exit(1);
      }

      if (L.corner_table)
        any_svd42_corner = true;
      else
        any_svd42_old = true;
    }

    if (any_svd42_corner && any_svd42_old) {
      std::cout << "ERROR: Mixed old/corner SVD4x2 table layouts are not "
                << "currently supported in one run." << std::endl;
      exit(1);
    }

    tabulate_4B_svd4x2_corner_kk = any_svd42_corner;

    std::vector<int> h_rank(nquads), h_ngrid_left(nquads), h_ngrid_right(nquads);
    std::vector<int> h_off_left(nquads), h_off_right(nquads);
    std::vector<int> h_canon_to_param(6 * nquads, 0);

    std::vector<double> h_r0_left(4 * nquads), h_dr_left(4 * nquads), h_invdr_left(4 * nquads);
    std::vector<double> h_r0_right(2 * nquads), h_dr_right(2 * nquads), h_invdr_right(2 * nquads);

    int total_left = 0;
    int total_right = 0;

    for (int q = 0; q < nquads; q++) {
      const int R = tab_4b_svd4x2_rank[q];
      h_rank[q] = R;

      const auto &L = tab_4b_svd4x2_left[q];
      const auto &Rt = tab_4b_svd4x2_right[q];

      if (L.rank != R || Rt.rank != R) {
        std::cout << "ERROR: SVD4x2 rank mismatch for quad " << q << std::endl;
        exit(1);
      }

      const int nL = infer_svd4_left_n_host((int)(L.block[0].size() / R), "SVD4x2", q);
      const int nR = infer_svd4_right_n_host((int)(Rt.block[0].size() / R), "SVD4x2", q);

      if (nL != L.ngrid || nR != Rt.ngrid) {
        std::cout << "ERROR: SVD4x2 grid mismatch for quad " << q << std::endl;
        exit(1);
      }

      h_ngrid_left[q] = nL;
      h_ngrid_right[q] = nR;
      h_off_left[q] = total_left;
      h_off_right[q] = total_right;

      for (int d = 0; d < 4; d++) {
        h_r0_left [4*q + d] = L.r0[d];
        h_dr_left [4*q + d] = L.dr[d];
        h_invdr_left[4*q + d] = L.invdr[d];
      }
      for (int d = 0; d < 2; d++) {
        h_r0_right [2*q + d] = Rt.r0[d];
        h_dr_right [2*q + d] = Rt.dr[d];
        h_invdr_right[2*q + d] = Rt.invdr[d];
      }

      for (int c = 0; c < 6; c++) {
        h_canon_to_param[6*q + c] =
          (tab_4b_svd4x2_canon_to_param[q].size() == 6)
          ? tab_4b_svd4x2_canon_to_param[q][c] : c;
      }

      total_left  += (int)L.block[0].size();
      total_right += (int)Rt.block[0].size();
    }

    std::vector<float> h_Lv(total_left), h_Ld0(total_left), h_Ld1(total_left), h_Ld2(total_left), h_Ld3(total_left);
    std::vector<float> h_Rv(total_right), h_Rd0(total_right), h_Rd1(total_right);

    for (int q = 0; q < nquads; q++) {
      const int offL = h_off_left[q];
      const int offR = h_off_right[q];

      const auto &L = tab_4b_svd4x2_left[q];
      const auto &Rt = tab_4b_svd4x2_right[q];

      for (int i = 0; i < (int)L.block[0].size(); i++) {
        h_Lv [offL + i] = L.block[0][i];
        h_Ld0[offL + i] = L.block[1][i];
        h_Ld1[offL + i] = L.block[2][i];
        h_Ld2[offL + i] = L.block[3][i];
        h_Ld3[offL + i] = L.block[4][i];
      }

      for (int i = 0; i < (int)Rt.block[0].size(); i++) {
        h_Rv [offR + i] = Rt.block[0][i];
        h_Rd0[offR + i] = Rt.block[1][i];
        h_Rd1[offR + i] = Rt.block[2][i];
      }
    }

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_rank, "chimesFF:tab4b_svd42_rank", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_ngrid_left, "chimesFF:tab4b_svd42_ngrid_left", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_ngrid_right, "chimesFF:tab4b_svd42_ngrid_right", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_offset_left, "chimesFF:tab4b_svd42_offset_left", nquads);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_offset_right, "chimesFF:tab4b_svd42_offset_right", nquads);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_r0_left,    "chimesFF:tab4b_svd42_r0_left",    nquads, 4);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_dr_left,    "chimesFF:tab4b_svd42_dr_left",    nquads, 4);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_invdr_left, "chimesFF:tab4b_svd42_invdr_left", nquads, 4);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_r0_right,    "chimesFF:tab4b_svd42_r0_right",    nquads, 2);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_dr_right,    "chimesFF:tab4b_svd42_dr_right",    nquads, 2);
    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_invdr_right, "chimesFF:tab4b_svd42_invdr_right", nquads, 2);

    LAMMPS_NS::MemKK::realloc_kokkos(d_tab4b_svd42_canon_to_param, "chimesFF:tab4b_svd42_canon_to_param", nquads, 6);

    d_tab4b_svd42_left_val = Kokkos::View<float*>("chimesFF:tab4b_svd42_left_val", total_left);
    d_tab4b_svd42_left_d0  = Kokkos::View<float*>("chimesFF:tab4b_svd42_left_d0", total_left);
    d_tab4b_svd42_left_d1  = Kokkos::View<float*>("chimesFF:tab4b_svd42_left_d1", total_left);
    d_tab4b_svd42_left_d2  = Kokkos::View<float*>("chimesFF:tab4b_svd42_left_d2", total_left);
    d_tab4b_svd42_left_d3  = Kokkos::View<float*>("chimesFF:tab4b_svd42_left_d3", total_left);

    d_tab4b_svd42_right_val = Kokkos::View<float*>("chimesFF:tab4b_svd42_right_val", total_right);
    d_tab4b_svd42_right_d0  = Kokkos::View<float*>("chimesFF:tab4b_svd42_right_d0", total_right);
    d_tab4b_svd42_right_d1  = Kokkos::View<float*>("chimesFF:tab4b_svd42_right_d1", total_right);

    auto h_rank_v  = Kokkos::create_mirror_view(d_tab4b_svd42_rank);
    auto h_ngL_v   = Kokkos::create_mirror_view(d_tab4b_svd42_ngrid_left);
    auto h_ngR_v   = Kokkos::create_mirror_view(d_tab4b_svd42_ngrid_right);
    auto h_offL_v  = Kokkos::create_mirror_view(d_tab4b_svd42_offset_left);
    auto h_offR_v  = Kokkos::create_mirror_view(d_tab4b_svd42_offset_right);

    auto h_r0L_v   = Kokkos::create_mirror_view(d_tab4b_svd42_r0_left);
    auto h_drL_v   = Kokkos::create_mirror_view(d_tab4b_svd42_dr_left);
    auto h_ivL_v   = Kokkos::create_mirror_view(d_tab4b_svd42_invdr_left);

    auto h_r0R_v   = Kokkos::create_mirror_view(d_tab4b_svd42_r0_right);
    auto h_drR_v   = Kokkos::create_mirror_view(d_tab4b_svd42_dr_right);
    auto h_ivR_v   = Kokkos::create_mirror_view(d_tab4b_svd42_invdr_right);

    auto h_map_v   = Kokkos::create_mirror_view(d_tab4b_svd42_canon_to_param);

    auto h_Lv_v    = Kokkos::create_mirror_view(d_tab4b_svd42_left_val);
    auto h_Ld0_v   = Kokkos::create_mirror_view(d_tab4b_svd42_left_d0);
    auto h_Ld1_v   = Kokkos::create_mirror_view(d_tab4b_svd42_left_d1);
    auto h_Ld2_v   = Kokkos::create_mirror_view(d_tab4b_svd42_left_d2);
    auto h_Ld3_v   = Kokkos::create_mirror_view(d_tab4b_svd42_left_d3);

    auto h_Rv_v    = Kokkos::create_mirror_view(d_tab4b_svd42_right_val);
    auto h_Rd0_v   = Kokkos::create_mirror_view(d_tab4b_svd42_right_d0);
    auto h_Rd1_v   = Kokkos::create_mirror_view(d_tab4b_svd42_right_d1);

    for (int q = 0; q < nquads; q++) {
      h_rank_v[q] = h_rank[q];
      h_ngL_v[q] = h_ngrid_left[q];
      h_ngR_v[q] = h_ngrid_right[q];
      h_offL_v[q] = h_off_left[q];
      h_offR_v[q] = h_off_right[q];

      for (int d = 0; d < 4; d++) {
        h_r0L_v(q,d) = h_r0_left[4*q + d];
        h_drL_v(q,d) = h_dr_left[4*q + d];
        h_ivL_v(q,d) = h_invdr_left[4*q + d];
      }
      for (int d = 0; d < 2; d++) {
        h_r0R_v(q,d) = h_r0_right[2*q + d];
        h_drR_v(q,d) = h_dr_right[2*q + d];
        h_ivR_v(q,d) = h_invdr_right[2*q + d];
      }
      for (int c = 0; c < 6; c++) h_map_v(q,c) = h_canon_to_param[6*q + c];
    }

    for (int i = 0; i < total_left; i++) {
      h_Lv_v[i]  = h_Lv[i];
      h_Ld0_v[i] = h_Ld0[i];
      h_Ld1_v[i] = h_Ld1[i];
      h_Ld2_v[i] = h_Ld2[i];
      h_Ld3_v[i] = h_Ld3[i];
    }
    for (int i = 0; i < total_right; i++) {
      h_Rv_v[i]  = h_Rv[i];
      h_Rd0_v[i] = h_Rd0[i];
      h_Rd1_v[i] = h_Rd1[i];
    }

    Kokkos::deep_copy(d_tab4b_svd42_rank, h_rank_v);
    Kokkos::deep_copy(d_tab4b_svd42_ngrid_left, h_ngL_v);
    Kokkos::deep_copy(d_tab4b_svd42_ngrid_right, h_ngR_v);
    Kokkos::deep_copy(d_tab4b_svd42_offset_left, h_offL_v);
    Kokkos::deep_copy(d_tab4b_svd42_offset_right, h_offR_v);
    Kokkos::deep_copy(d_tab4b_svd42_r0_left, h_r0L_v);
    Kokkos::deep_copy(d_tab4b_svd42_dr_left, h_drL_v);
    Kokkos::deep_copy(d_tab4b_svd42_invdr_left, h_ivL_v);
    Kokkos::deep_copy(d_tab4b_svd42_r0_right, h_r0R_v);
    Kokkos::deep_copy(d_tab4b_svd42_dr_right, h_drR_v);
    Kokkos::deep_copy(d_tab4b_svd42_invdr_right, h_ivR_v);
    Kokkos::deep_copy(d_tab4b_svd42_canon_to_param, h_map_v);

    Kokkos::deep_copy(d_tab4b_svd42_left_val, h_Lv_v);
    Kokkos::deep_copy(d_tab4b_svd42_left_d0, h_Ld0_v);
    Kokkos::deep_copy(d_tab4b_svd42_left_d1, h_Ld1_v);
    Kokkos::deep_copy(d_tab4b_svd42_left_d2, h_Ld2_v);
    Kokkos::deep_copy(d_tab4b_svd42_left_d3, h_Ld3_v);

    Kokkos::deep_copy(d_tab4b_svd42_right_val, h_Rv_v);
    Kokkos::deep_copy(d_tab4b_svd42_right_d0, h_Rd0_v);
    Kokkos::deep_copy(d_tab4b_svd42_right_d1, h_Rd1_v);

    c_tab4b_svd42_rank = d_tab4b_svd42_rank;
    c_tab4b_svd42_ngrid_left = d_tab4b_svd42_ngrid_left;
    c_tab4b_svd42_ngrid_right = d_tab4b_svd42_ngrid_right;
    c_tab4b_svd42_offset_left = d_tab4b_svd42_offset_left;
    c_tab4b_svd42_offset_right = d_tab4b_svd42_offset_right;
    c_tab4b_svd42_r0_left = d_tab4b_svd42_r0_left;
    c_tab4b_svd42_dr_left = d_tab4b_svd42_dr_left;
    c_tab4b_svd42_invdr_left = d_tab4b_svd42_invdr_left;
    c_tab4b_svd42_r0_right = d_tab4b_svd42_r0_right;
    c_tab4b_svd42_dr_right = d_tab4b_svd42_dr_right;
    c_tab4b_svd42_invdr_right = d_tab4b_svd42_invdr_right;
    c_tab4b_svd42_canon_to_param = d_tab4b_svd42_canon_to_param;

    if (rank == 0)
      std::cout << "chimesFF/kk: packed 4B SVD4x2 tabulation" << std::endl;
  }
#endif

  // poly_orders

  size = poly_orders.size();

  for (int i = 0; i < size; i++)
    d_poly_orders[i] = poly_orders[i];


  // morse_var

  size = morse_var.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_morse_var,"chimesFF:morse_var",size);

  auto h_morse_var = Kokkos::create_mirror_view(d_morse_var);

  for (int i = 0; i < size; i++)
    h_morse_var[i] = morse_var[i];

  Kokkos::deep_copy(d_morse_var,h_morse_var);
  c_morse_var = d_morse_var;


  // penalty_params

  d_penalty_params[0] = penalty_params[0];
  d_penalty_params[1] = penalty_params[1];


  // ncoeffs_2b

  size = ncoeffs_2b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_2b,"chimesFF:ncoeffs_2b",size);

  auto h_ncoeffs_2b = Kokkos::create_mirror_view(d_ncoeffs_2b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_2b[i] = ncoeffs_2b[i];

  Kokkos::deep_copy(d_ncoeffs_2b,h_ncoeffs_2b);
  c_ncoeffs_2b = d_ncoeffs_2b;


  // chimes_2b_pows

  size = chimes_2b_pows.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_2b_pows[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_pows,"chimesFF:chimes_2b_pows",size,max_j);

  auto h_chimes_2b_pows = Kokkos::create_mirror_view(d_chimes_2b_pows);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_2b_pows[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_2b_pows(i,j) = chimes_2b_pows[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_2b_pows,h_chimes_2b_pows);
  c_chimes_2b_pows = d_chimes_2b_pows;

  // chimes_2b_params

  size = chimes_2b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_2b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_params,"chimesFF:chimes_2b_params",size,max_j);

  auto h_chimes_2b_params = Kokkos::create_mirror_view(d_chimes_2b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_2b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_2b_params(i,j) = chimes_2b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_2b_params,h_chimes_2b_params);
  c_chimes_2b_params = d_chimes_2b_params;

  // chimes_2b_cutoff

  size = chimes_2b_cutoff.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_cutoff,"chimesFF:chimes_2b_cutoff",size);

  auto h_chimes_2b_cutoff = Kokkos::create_mirror_view(d_chimes_2b_cutoff);

  for (int i = 0; i < size; i++) {
    h_chimes_2b_cutoff(i,0) = chimes_2b_cutoff[i][0];
    h_chimes_2b_cutoff(i,1) = chimes_2b_cutoff[i][1];
  }

  Kokkos::deep_copy(d_chimes_2b_cutoff,h_chimes_2b_cutoff);
  c_chimes_2b_cutoff = d_chimes_2b_cutoff;

  // ncoeffs_3b

  size = ncoeffs_3b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_3b,"chimesFF:ncoeffs_3b",size);

  auto h_ncoeffs_3b = Kokkos::create_mirror_view(d_ncoeffs_3b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_3b[i] = ncoeffs_3b[i];

  Kokkos::deep_copy(d_ncoeffs_3b,h_ncoeffs_3b);
  c_ncoeffs_3b = d_ncoeffs_3b;


  // chimes_3b_powers

  size = chimes_3b_powers.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_powers[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_powers[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_powers,"chimesFF:chimes_3b_powers",size,max_j,max_k);

  auto h_chimes_3b_powers = Kokkos::create_mirror_view(d_chimes_3b_powers);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_powers(i,j,k) = chimes_3b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_powers,h_chimes_3b_powers);
  c_chimes_3b_powers = d_chimes_3b_powers;


  // chimes_3b_params

  size = chimes_3b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_3b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_params,"chimesFF:chimes_3b_params",size,max_j);

  auto h_chimes_3b_params = Kokkos::create_mirror_view(d_chimes_3b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_3b_params(i,j) = chimes_3b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_3b_params,h_chimes_3b_params);
  c_chimes_3b_params = d_chimes_3b_params;


  // chimes_3b_cutoff

  size = chimes_3b_cutoff.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_cutoff[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_cutoff[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // last 2 indices are switched from the CPU code

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_cutoff,"chimesFF:chimes_3b_cutoff",size,max_k,max_j);

  auto h_chimes_3b_cutoff = Kokkos::create_mirror_view(d_chimes_3b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_cutoff(i,k,j) = chimes_3b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_cutoff,h_chimes_3b_cutoff);
  c_chimes_3b_cutoff = d_chimes_3b_cutoff;


  // ncoeffs_4b

  size = ncoeffs_4b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_4b,"chimesFF:ncoeffs_4b",size);

  auto h_ncoeffs_4b = Kokkos::create_mirror_view(d_ncoeffs_4b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_4b[i] = ncoeffs_4b[i];

  Kokkos::deep_copy(d_ncoeffs_4b,h_ncoeffs_4b);
  c_ncoeffs_4b = d_ncoeffs_4b;


  // chimes_4b_powers

  size = chimes_4b_powers.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_powers[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_powers[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_powers,"chimesFF:chimes_4b_powers",size,max_j,max_k);

  auto h_chimes_4b_powers = Kokkos::create_mirror_view(d_chimes_4b_powers);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_powers(i,j,k) = chimes_4b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_powers,h_chimes_4b_powers);
  c_chimes_4b_powers = d_chimes_4b_powers;


  // chimes_4b_params

  size = chimes_4b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_4b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_params,"chimesFF:chimes_4b_params",size,max_j);

  auto h_chimes_4b_params = Kokkos::create_mirror_view(d_chimes_4b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_4b_params(i,j) = chimes_4b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_4b_params,h_chimes_4b_params);
  c_chimes_4b_params = d_chimes_4b_params;


  // chimes_4b_cutoff

  size = chimes_4b_cutoff.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_cutoff[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_cutoff[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // last 2 indices are switched from the CPU code

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_cutoff,"chimesFF:chimes_4b_cutoff",size,max_k,max_j);

  auto h_chimes_4b_cutoff = Kokkos::create_mirror_view(d_chimes_4b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_cutoff(i,k,j) = chimes_4b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_cutoff,h_chimes_4b_cutoff);
  c_chimes_4b_cutoff = d_chimes_4b_cutoff;


  // energy_offsets

  size = energy_offsets.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_energy_offsets,"chimesFF:energy_offsets",size);

  auto h_energy_offsets = Kokkos::create_mirror_view(d_energy_offsets);

  for (int i = 0; i < size; i++)
    h_energy_offsets[i] = energy_offsets[i];

  Kokkos::deep_copy(d_energy_offsets,h_energy_offsets);
  c_energy_offsets = d_energy_offsets;


  // atom_int_pair_map

  size = atom_int_pair_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_pair_map,"chimesFF:atom_int_pair_map",size);

  auto h_atom_int_pair_map = Kokkos::create_mirror_view(d_atom_int_pair_map);

  for (int i = 0; i < size; i++)
    h_atom_int_pair_map[i] = atom_int_pair_map[i];

  Kokkos::deep_copy(d_atom_int_pair_map,h_atom_int_pair_map);
  c_atom_int_pair_map = d_atom_int_pair_map;


  // atom_int_trip_map

  size = atom_int_trip_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_trip_map,"chimesFF:",size);

  auto h_atom_int_trip_map = Kokkos::create_mirror_view(d_atom_int_trip_map);

  for (int i = 0; i < size; i++)
    h_atom_int_trip_map[i] = atom_int_trip_map[i];

  Kokkos::deep_copy(d_atom_int_trip_map,h_atom_int_trip_map);
  c_atom_int_trip_map = d_atom_int_trip_map;


  // atom_int_quad_map

  size = atom_int_quad_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_quad_map,"chimesFF:",size);

  auto h_atom_int_quad_map = Kokkos::create_mirror_view(d_atom_int_quad_map);

  for (int i = 0; i < size; i++)
    h_atom_int_quad_map[i] = atom_int_quad_map[i];

  Kokkos::deep_copy(d_atom_int_quad_map,h_atom_int_quad_map);
  c_atom_int_quad_map = d_atom_int_quad_map;

  release_host_tables();
}

/* ---------------------------------------------------------------------- */

template<class T>
static inline void chimes_kk_free_host(T &v)
{
  T().swap(v);
}

template<class DeviceType>
void chimesFFKokkos<DeviceType>::release_host_tables()
{
  // The kernels read only the packed device views, so the host copies of
  // the tables are no longer needed.  Freeing them matters for speed: this
  // object is a member of the pair style, which Kokkos copies by value at
  // every kernel launch, and each std::vector left here would be
  // deep-copied on the host for every launch (tens of MB for large CP
  // ranks or 3D 3B tables).
#ifdef TABULATION
  if (tabulate_2B) {
    chimes_kk_free_host(tab_r);
    chimes_kk_free_host(tab_e);
    chimes_kk_free_host(tab_f);
  }

  if (tabulate_3B) {
    chimes_kk_free_host(tab_rij_3B);
    chimes_kk_free_host(tab_rik_3B);
    chimes_kk_free_host(tab_rjk_3B);
    chimes_kk_free_host(tab_e_3B);
    chimes_kk_free_host(tab_f_ij_3B);
    chimes_kk_free_host(tab_f_ik_3B);
    chimes_kk_free_host(tab_f_jk_3B);
  }

  if (tabulate_4B_svd3x3) {
    chimes_kk_free_host(tab_4b_svd3x3_left);
    chimes_kk_free_host(tab_4b_svd3x3_right);
  }

  if (tabulate_4B_svd4x2) {
    chimes_kk_free_host(tab_4b_svd4x2_left);
    chimes_kk_free_host(tab_4b_svd4x2_right);
  }

  if (tabulate_4B_cp)
    chimes_kk_free_host(tab_4b_cp_slot);

  if (tabulate_4B_cp_direct)
    chimes_kk_free_host(tab_4b_cp_direct_factor);

  if (tabulate_3B_cp)
    chimes_kk_free_host(tab_3b_cp);
#endif
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::set_polys_out_of_range(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, KK_FLOAT x, int poly_order, KK_FLOAT inner_cutoff, KK_FLOAT exprlen, KK_FLOAT dx_dr) const
{
  //  Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
  //  Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
  //	
  //  The derivative Tnd is continuously set to zero inside the cutoff.
  //  The exponential smoothing distance is set to ChimesFF::inner_smooth_distance.
  //  x, exprlen, and dx_dr are evaluated at the inner cutoff.
  //	
  //  dx is the pair distance, which is assumed to be less than inner_cutoff.
  Tn[0] = 1.0;
  Tn[1] = x;

  // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

  Tnd[0] = 1.0;
  Tnd[1] = 2.0 * x;

  // Use recursion to set up the higher n-value Tn and Tnd's
  for (int i = 2; i <= poly_order; i++) {
    Tn[i] = 2.0 * x * Tn[i-1] - Tn[i-2];
    Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
  }

  // Now multiply by n to convert Tnd's to actual derivatives of Tn

  for (int i = poly_order; i >= 1; i--)
    Tnd[i] = i * dx_dr * Tnd[i-1];

  Tnd[0] = 0.0;

  // Exponential damping of the derivative.
  const KK_FLOAT damp_fac = exp((dx-inner_cutoff) / inner_smooth_distance);

  // Correct Tn outside of the range using the damping factor.
  for (int i = 0 ; i <= poly_order ; i++) {
    Tn[i] += inner_smooth_distance * (damp_fac-1.0)  * Tnd[i];
    Tnd[i] *= damp_fac;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT chimesFFKokkos<DeviceType>::dr2_3B(const KK_FLOAT *dr2, int i, int j, int k, int l) const
{
  // Access the dr2 distance tensor for a 3 body interaction

  return(dr2[i*CHDIM*3*CHDIM + j*3*CHDIM + k*CHDIM + l]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT chimesFFKokkos<DeviceType>::dr2_4B(const KK_FLOAT *dr2, int i, int j, int k, int l) const
{
  // Access the dr2 distance tensor for a 4 body interaction

  return(dr2[i*CHDIM*6*CHDIM + j*6*CHDIM + k*CHDIM + l]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::init_distance_tensor(KK_FLOAT *dr2, const KK_FLOAT* dr, int npairs) const
{
  for (int i = 0; i < npairs; i++ )
    for (int j = 0; j < CHDIM; j++ )
      for (int k = 0; k < npairs; k++ )
        for (int l = 0; l < CHDIM; l++ )
          dr2[i* CHDIM * npairs * CHDIM + j * npairs * CHDIM + k * CHDIM + l] = dr[i*CHDIM+j] * dr[k*CHDIM+l];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_1B(const int typ_idx, KK_FLOAT & energy ) const
{
  // Compute 1b (input: a single atom type index... outputs (updates) energy

  energy += c_energy_offsets[typ_idx];
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar;
  compute_2B(dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar) const
{
  // Compute 2b (input: 2 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx: Scalar (pair distance)
  // dr: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]  *note
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for calculation.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force is a packed array of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  KK_FLOAT fcut;
  KK_FLOAT fcutderiv;

  // Use references for readability

  KK_FLOAT Tn[MAX_2B_POLY];
  KK_FLOAT Tnd[MAX_2B_POLY];

  const int pair_idx = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);

  //if (dx >= c_chimes_2b_cutoff(pair_idx,1)) return;

  set_cheby_polys(Tn, Tnd, dx, c_morse_var[pair_idx], c_chimes_2b_cutoff(pair_idx,0), c_chimes_2b_cutoff(pair_idx,1), d_poly_orders[0]);

  get_fcut(dx, c_chimes_2b_cutoff(pair_idx,1), fcut, fcutderiv);

  KK_FLOAT poly, dpoly_dx;

  poly_2B(poly, dpoly_dx, c_ncoeffs_2b[pair_idx], pair_idx, Tn, Tnd);

  //const KK_FLOAT dx_inv = (dx > 0.0 ) ? 1.0 / dx : 1e20;

  if (eflag)
    energy += poly * fcut;

  force_scalar = (fcut * dpoly_dx + fcutderiv * poly) / dx;

  force[0] += force_scalar * dr[0];
  force[1] += force_scalar * dr[1];
  force[2] += force_scalar * dr[2];

  force[CHDIM+0] -= force_scalar * dr[0];
  force[CHDIM+1] -= force_scalar * dr[1];
  force[CHDIM+2] -= force_scalar * dr[2];

  // xx xy xz yy yz zz
  // 0  1  2  3  4  5

  // xx xy xz yx yy yz zx zy zz
  // 0  1  2  3  4  5  6  7  8
  // *           *           *

  if (vflag) {
    stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
    stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
    stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
    stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
    stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
    stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
  }

  KK_FLOAT E_penalty = 0.0;
  get_penalty(dx, pair_idx, E_penalty, force_scalar);

  if (E_penalty > 0.0 )
  {
    if (eflag)
      energy += E_penalty;

    force_scalar /= dx;

    // Note: force_scalar is negative (LEF) 7/30/21

    force[0] += force_scalar * dr[0];
    force[1] += force_scalar * dr[1];
    force[2] += force_scalar * dr[2];

    force[CHDIM+0] -= force_scalar * dr[0];
    force[CHDIM+1] -= force_scalar * dr[1];
    force[CHDIM+2] -= force_scalar * dr[2];

    // Update stress according to penalty force. (LEF) 07/30/21

    if (vflag) {
      stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
      stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
      stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
      stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
      stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
      stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
    }
  }
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[3];
  compute_3B(dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for 3-body interactions.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  constexpr int natoms = 3;                   // Number of atoms in an interaction set
  constexpr int npairs = natoms*(natoms-1)/2; // Number of pairs in an interaction set

  KK_FLOAT Tn_ij[MAX_3B_POLY];
  KK_FLOAT Tn_ik[MAX_3B_POLY];
  KK_FLOAT Tn_jk[MAX_3B_POLY];

  KK_FLOAT Tnd_ij[MAX_3B_POLY];
  KK_FLOAT Tnd_ik[MAX_3B_POLY];
  KK_FLOAT Tnd_jk[MAX_3B_POLY];

  // Avoid allocating vector quantities.  Heap memory allocation is slow on the GPU.
  // fixed-length C arrays are allocated on the stack

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];
  KK_FLOAT deriv[npairs];

  const int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
  const int tripidx = c_atom_int_trip_map[type_idx];

  //if (tripidx < 0) // Skipping an excluded interaction
  //  return;

  // Check whether cutoffs are within allowed ranges
  //auto c_mapped_pair_idx = c_pair_int_trip_map[type_idx];

  const KK_FLOAT cutoff_00 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,0),0);
  const KK_FLOAT cutoff_0 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,0),1);

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  const KK_FLOAT cutoff_01 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,1),0);
  const KK_FLOAT cutoff_1 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,1),1);

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  const KK_FLOAT cutoff_02 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,2),0);
  const KK_FLOAT cutoff_2 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,2),1);

  //if (dx[2] >= cutoff_2) // jk
  //  return;

 const int pair_type_1 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
 const int pair_type_2 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
 const int pair_type_3 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
 const int order = d_poly_orders[1];

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  if (vflag)
    init_distance_tensor(dr2, dr, npairs);
#endif

  // Set up the polynomials

  set_cheby_polys(Tn_ij, Tnd_ij, dx[0], c_morse_var[pair_type_1], cutoff_00, cutoff_0, order);
  set_cheby_polys(Tn_ik, Tnd_ik, dx[1], c_morse_var[pair_type_2], cutoff_01, cutoff_1, order);
  set_cheby_polys(Tn_jk, Tnd_jk, dx[2], c_morse_var[pair_type_3], cutoff_02, cutoff_2, order);

  // Set up the smoothing functions

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  const KK_FLOAT fcut_all = fcut[0] * fcut[1] * fcut[2];

  KK_FLOAT poly, dpoly_dx[npairs];

  // Start the force/stress/energy calculation

  if (true) {
    poly_3B(poly, dpoly_dx, c_ncoeffs_3b[tripidx], tripidx, type_idx,
            Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
  } else {

    // JIT evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++)
      inv_mapped_pair[c_pair_int_trip_map(type_idx,j)] = j;

    KK_FLOAT* Tn[npairs];
    KK_FLOAT* Tnd[npairs];

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij[0];
          Tnd[j] = &Tnd_ij[0];
          break;
        case 1:
          Tn[j] = &Tn_ik[0];
          Tnd[j] = &Tnd_ik[0];
          break;
        case 2:
          Tn[j] = &Tn_jk[0];
          Tnd[j] = &Tnd_jk[0];
          break;
        default:
          Kokkos::abort("Bad inverse pair mapping found");
      }
    }

    poly_3B_dense(poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], c_ncoeffs_3b[tripidx], tripidx,
                  Tn[0], Tn[1], Tn[2], Tnd[0], Tnd[1], Tnd[2]);
  }

  if (eflag)
    energy += poly * fcut_all;

  force_scalar[0] = (fcut_all * dpoly_dx[0] + fcutderiv[0] * fcut[1] * fcut[2] * poly) / dx[0];
  force_scalar[1] = (fcut_all * dpoly_dx[1] + fcutderiv[1] * fcut[0] * fcut[2] * poly) / dx[1];
  force_scalar[2] = (fcut_all * dpoly_dx[2] + fcutderiv[2] * fcut[0] * fcut[1] * poly) / dx[2];

  const KK_FLOAT &fscalar_0 = force_scalar[0];
  const KK_FLOAT &fscalar_1 = force_scalar[1];
  const KK_FLOAT &fscalar_2 = force_scalar[2];

  // Accumulate forces/stresses on/from the ij pair

  force[0] += fscalar_0 * dr[0];
  force[1] += fscalar_0 * dr[1];
  force[2] += fscalar_0 * dr[2];

  force[CHDIM+0] -= fscalar_0 * dr[0];
  force[CHDIM+1] -= fscalar_0 * dr[1];
  force[CHDIM+2] -= fscalar_0 * dr[2];

  // dr2_3B looks like a function call, but the optimizer should remove it entirely
  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    // New stress code

    stress[0] -= fscalar_0 * dr2_3B(dr2,0,0,0,0); // xx tensor component
    stress[1] -= fscalar_0 * dr2_3B(dr2,0,0,0,1); // xy tensor component
    stress[2] -= fscalar_0 * dr2_3B(dr2,0,0,0,2); // xz tensor component
    stress[3] -= fscalar_0 * dr2_3B(dr2,0,1,0,1); // yy tensor component
    stress[4] -= fscalar_0 * dr2_3B(dr2,0,1,0,2); // yz tensor component
    stress[5] -= fscalar_0 * dr2_3B(dr2,0,2,0,2); // zz tensor component

#else
    stress[0] -= fscalar_0 * dr[0] * dr[0]; // xx tensor component
    stress[1] -= fscalar_0 * dr[0] * dr[1]; // xy tensor component
    stress[2] -= fscalar_0 * dr[0] * dr[2]; // xz tensor component
    stress[3] -= fscalar_0 * dr[1] * dr[1]; // yy tensor component
    stress[4] -= fscalar_0 * dr[1] * dr[2]; // yz tensor component
    stress[5] -= fscalar_0 * dr[2] * dr[2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0] += fscalar_1 * dr[CHDIM+0];
  force[1] += fscalar_1 * dr[CHDIM+1];
  force[2] += fscalar_1 * dr[CHDIM+2];

  force[2*CHDIM+0] -= fscalar_1 * dr[CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_1 * dr2_3B(dr2,1,0,1,0); // xx tensor component
    stress[1] -= fscalar_1 * dr2_3B(dr2,1,0,1,1); // xy tensor component
    stress[2] -= fscalar_1 * dr2_3B(dr2,1,0,1,2); // xz tensor component
    stress[3] -= fscalar_1 * dr2_3B(dr2,1,1,1,1); // yy tensor component
    stress[4] -= fscalar_1 * dr2_3B(dr2,1,1,1,2); // yz tensor component
    stress[5] -= fscalar_1 * dr2_3B(dr2,1,2,1,2); // zz tensor component
#else
    stress[0] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_1 * dr[CHDIM+2] * dr[CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jk pair

  force[CHDIM+0] += fscalar_2 * dr[2*CHDIM+0];
  force[CHDIM+1] += fscalar_2 * dr[2*CHDIM+1];
  force[CHDIM+2] += fscalar_2 * dr[2*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_2 * dr2_3B(dr2,2,0,2,0); // xx tensor component
    stress[1] -= fscalar_2 * dr2_3B(dr2,2,0,2,1); // xy tensor component
    stress[2] -= fscalar_2 * dr2_3B(dr2,2,0,2,2); // xz tensor component
    stress[3] -= fscalar_2 * dr2_3B(dr2,2,1,2,1); // yy tensor component
    stress[4] -= fscalar_2 * dr2_3B(dr2,2,1,2,2); // yz tensor component
    stress[5] -= fscalar_2 * dr2_3B(dr2,2,2,2,2); // zz tensor component
#else
    stress[0] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_2 * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component
#endif
  }
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT chimesFFKokkos<DeviceType>::interp_tab2b_linear(int pair_idx, KK_FLOAT r, const float *tab) const
{
  const int n = c_tab2b_n[pair_idx];
  const KK_FLOAT r0 = c_tab2b_r0[pair_idx];
  const KK_FLOAT dr = c_tab2b_dr[pair_idx];
  const KK_FLOAT invdr = c_tab2b_invdr[pair_idx];
  const int off = c_tab2b_offset[pair_idx];

  int i = (int)((r - r0) * invdr);
  if (i < 0) i = 0;
  if (i > n - 2) i = n - 2;

  KK_FLOAT x0 = r0 + i * dr;
  KK_FLOAT t = (r - x0) * invdr;
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;

  const KK_FLOAT y0 = (KK_FLOAT)tab[off + i];
  const KK_FLOAT y1 = (KK_FLOAT)tab[off + i + 1];

  return (1.0 - t) * y0 + t * y1;
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B_tab(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs,
                                                KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const
{
  const int pair_idx = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
  if (dx >= c_chimes_2b_cutoff(pair_idx,1)) return;

  const KK_FLOAT e2 = interp_tab2b_linear(pair_idx, dx, d_tab2b_e.data());
  const KK_FLOAT fs = interp_tab2b_linear(pair_idx, dx, d_tab2b_f.data());

  if (eflag) energy += e2;

  force[0] += fs * dr[0];
  force[1] += fs * dr[1];
  force[2] += fs * dr[2];

  force[CHDIM+0] -= fs * dr[0];
  force[CHDIM+1] -= fs * dr[1];
  force[CHDIM+2] -= fs * dr[2];

  if (vflag) {
    stress[0] -= fs * dr[0] * dr[0];
    stress[1] -= fs * dr[0] * dr[1];
    stress[2] -= fs * dr[0] * dr[2];
    stress[3] -= fs * dr[1] * dr[1];
    stress[4] -= fs * dr[1] * dr[2];
    stress[5] -= fs * dr[2] * dr[2];
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::interp_tab3b_linear(
    int tripidx, const KK_FLOAT *rquery,
    KK_FLOAT &e, KK_FLOAT &d0, KK_FLOAT &d1, KK_FLOAT &d2) const
{
  const int n = c_tab3b_n[tripidx];
  const KK_FLOAT r0 = c_tab3b_r0[tripidx];
  const KK_FLOAT dr = c_tab3b_dr[tripidx];
  const KK_FLOAT invdr = c_tab3b_invdr[tripidx];
  const int off = c_tab3b_offset[tripidx];

  int i0 = (int)((rquery[0] - r0) * invdr);
  int i1 = (int)((rquery[1] - r0) * invdr);
  int i2 = (int)((rquery[2] - r0) * invdr);

  if (i0 < 0) i0 = 0;
  if (i1 < 0) i1 = 0;
  if (i2 < 0) i2 = 0;
  if (i0 > n-2) i0 = n-2;
  if (i1 > n-2) i1 = n-2;
  if (i2 > n-2) i2 = n-2;

  const KK_FLOAT x0 = r0 + i0 * dr;
  const KK_FLOAT x1 = r0 + i1 * dr;
  const KK_FLOAT x2 = r0 + i2 * dr;

  KK_FLOAT t0 = (rquery[0] - x0) * invdr;
  KK_FLOAT t1 = (rquery[1] - x1) * invdr;
  KK_FLOAT t2 = (rquery[2] - x2) * invdr;

  if (t0 < 0.0) t0 = 0.0; if (t0 > 1.0) t0 = 1.0;
  if (t1 < 0.0) t1 = 0.0; if (t1 > 1.0) t1 = 1.0;
  if (t2 < 0.0) t2 = 0.0; if (t2 > 1.0) t2 = 1.0;

  const KK_FLOAT a0 = 1.0 - t0, b0 = t0;
  const KK_FLOAT a1 = 1.0 - t1, b1 = t1;
  const KK_FLOAT a2 = 1.0 - t2, b2 = t2;

  const int s0 = n*n;
  const int s1 = n;

  const int idx000 = off + i0*s0 + i1*s1 + i2;
  const int idx001 = idx000 + 1;
  const int idx010 = idx000 + s1;
  const int idx011 = idx010 + 1;
  const int idx100 = idx000 + s0;
  const int idx101 = idx100 + 1;
  const int idx110 = idx100 + s1;
  const int idx111 = idx110 + 1;

  const KK_FLOAT w000 = a0*a1*a2;
  const KK_FLOAT w001 = a0*a1*b2;
  const KK_FLOAT w010 = a0*b1*a2;
  const KK_FLOAT w011 = a0*b1*b2;
  const KK_FLOAT w100 = b0*a1*a2;
  const KK_FLOAT w101 = b0*a1*b2;
  const KK_FLOAT w110 = b0*b1*a2;
  const KK_FLOAT w111 = b0*b1*b2;

  const float *E  = d_tab3b_e.data();
  const float *F0 = d_tab3b_f0.data();
  const float *F1 = d_tab3b_f1.data();
  const float *F2 = d_tab3b_f2.data();

  e =
      w000*(KK_FLOAT)E[idx000] + w001*(KK_FLOAT)E[idx001] +
      w010*(KK_FLOAT)E[idx010] + w011*(KK_FLOAT)E[idx011] +
      w100*(KK_FLOAT)E[idx100] + w101*(KK_FLOAT)E[idx101] +
      w110*(KK_FLOAT)E[idx110] + w111*(KK_FLOAT)E[idx111];

  d0 =
      w000*(KK_FLOAT)F0[idx000] + w001*(KK_FLOAT)F0[idx001] +
      w010*(KK_FLOAT)F0[idx010] + w011*(KK_FLOAT)F0[idx011] +
      w100*(KK_FLOAT)F0[idx100] + w101*(KK_FLOAT)F0[idx101] +
      w110*(KK_FLOAT)F0[idx110] + w111*(KK_FLOAT)F0[idx111];

  d1 =
      w000*(KK_FLOAT)F1[idx000] + w001*(KK_FLOAT)F1[idx001] +
      w010*(KK_FLOAT)F1[idx010] + w011*(KK_FLOAT)F1[idx011] +
      w100*(KK_FLOAT)F1[idx100] + w101*(KK_FLOAT)F1[idx101] +
      w110*(KK_FLOAT)F1[idx110] + w111*(KK_FLOAT)F1[idx111];

  d2 =
      w000*(KK_FLOAT)F2[idx000] + w001*(KK_FLOAT)F2[idx001] +
      w010*(KK_FLOAT)F2[idx010] + w011*(KK_FLOAT)F2[idx011] +
      w100*(KK_FLOAT)F2[idx100] + w101*(KK_FLOAT)F2[idx101] +
      w110*(KK_FLOAT)F2[idx110] + w111*(KK_FLOAT)F2[idx111];
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs,
                                                KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const
{
  constexpr int npairs = 3;

  const int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
  const int tripidx = c_atom_int_trip_map[type_idx];
  if (tripidx < 0) return;

  const int pmap0 = c_pair_int_trip_map(type_idx,0);
  const int pmap1 = c_pair_int_trip_map(type_idx,1);
  const int pmap2 = c_pair_int_trip_map(type_idx,2);

  if (dx[0] >= c_chimes_3b_cutoff(tripidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_3b_cutoff(tripidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_3b_cutoff(tripidx,pmap2,1)) return;

  int param_to_runtime[3];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;

  int canon_to_runtime[3];
  for (int c = 0; c < 3; c++) {
    const int pslot = c_tab3b_canon_to_param(tripidx,c);
    canon_to_runtime[c] = param_to_runtime[pslot];
  }

  KK_FLOAT rq[3];
  rq[0] = dx[canon_to_runtime[0]];
  rq[1] = dx[canon_to_runtime[1]];
  rq[2] = dx[canon_to_runtime[2]];

  KK_FLOAT e3, dc0, dc1, dc2;
  interp_tab3b_linear(tripidx, rq, e3, dc0, dc1, dc2);

  if (eflag) energy += e3;

  KK_FLOAT fs[3];
  fs[canon_to_runtime[0]] = dc0;
  fs[canon_to_runtime[1]] = dc1;
  fs[canon_to_runtime[2]] = dc2;

  // ij
  force[0] += fs[0] * dr[0];
  force[1] += fs[0] * dr[1];
  force[2] += fs[0] * dr[2];
  force[CHDIM+0] -= fs[0] * dr[0];
  force[CHDIM+1] -= fs[0] * dr[1];
  force[CHDIM+2] -= fs[0] * dr[2];

  // ik
  force[0] += fs[1] * dr[CHDIM+0];
  force[1] += fs[1] * dr[CHDIM+1];
  force[2] += fs[1] * dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1] * dr[CHDIM+0];
  force[2*CHDIM+1] -= fs[1] * dr[CHDIM+1];
  force[2*CHDIM+2] -= fs[1] * dr[CHDIM+2];

  // jk
  force[CHDIM+0] += fs[2] * dr[2*CHDIM+0];
  force[CHDIM+1] += fs[2] * dr[2*CHDIM+1];
  force[CHDIM+2] += fs[2] * dr[2*CHDIM+2];
  force[2*CHDIM+0] -= fs[2] * dr[2*CHDIM+0];
  force[2*CHDIM+1] -= fs[2] * dr[2*CHDIM+1];
  force[2*CHDIM+2] -= fs[2] * dr[2*CHDIM+2];

  if (vflag) {
    for (int p = 0; p < 3; p++) {
      stress[0] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+0];
      stress[1] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+1];
      stress[2] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+2];
      stress[3] -= fs[p] * dr[p*CHDIM+1] * dr[p*CHDIM+1];
      stress[4] -= fs[p] * dr[p*CHDIM+1] * dr[p*CHDIM+2];
      stress[5] -= fs[p] * dr[p*CHDIM+2] * dr[p*CHDIM+2];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B_cp_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const
{
  // E_3B = sum_q X_0q(r_0) X_1q(r_1) X_2q(r_2) over the canonical slots of the
  // triplet type, each X_cq read from a CP 1D table (Kokkos version of the CPU
  // chimesFF::compute_3B_cp_tab).

  const int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
  const int tripidx = c_atom_int_trip_map[type_idx];
  if (tripidx < 0) return;

  const int pmap0 = c_pair_int_trip_map(type_idx,0);
  const int pmap1 = c_pair_int_trip_map(type_idx,1);
  const int pmap2 = c_pair_int_trip_map(type_idx,2);

  if (dx[0] >= c_chimes_3b_cutoff(tripidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_3b_cutoff(tripidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_3b_cutoff(tripidx,pmap2,1)) return;

  // Rank 0 only for triplet types without 3B terms (checked when packing)
  const int R = c_tab3b_cp_rank[tripidx];
  if (R <= 0) return;

  int param_to_runtime[3];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;

  int canon_to_runtime[3];

  const chimes_tab_real *V = d_tab3b_cp_vd.data();
  const chimes_tab_real *row[3];
  KK_FLOAT wv[3][4];
  KK_FLOAT wd[3][4];

  for (int c = 0; c < 3; c++) {
    const int pslot = c_tab3b_cp_canon_to_param(tripidx,c);
    canon_to_runtime[c] = param_to_runtime[pslot];

    const int i = cp1d_weights(dx[canon_to_runtime[c]], c_tab3b_cp_r0(tripidx,c), c_tab3b_cp_dr(tripidx,c), c_tab3b_cp_invdr(tripidx,c), c_tab3b_cp_ngrid(tripidx,c), c_chimes_3b_cutoff(tripidx,pslot,1), wv[c], wd[c]);

    row[c] = V + c_tab3b_cp_offset(tripidx,c) + (size_t)i * (size_t)(2 * R);
  }

  KK_FLOAT e3 = 0.0;
  KK_FLOAT g0 = 0.0, g1 = 0.0, g2 = 0.0;

  for (int q = 0; q < R; q++) {
    KK_FLOAT X[3];
    KK_FLOAT D[3];

    for (int c = 0; c < 3; c++) {
      const chimes_tab_real *p = row[c];

      const KK_FLOAT y0 = p[q];
      const KK_FLOAT m0 = p[R + q];
      const KK_FLOAT y1 = p[2*R + q];
      const KK_FLOAT m1 = p[3*R + q];

      X[c] = wv[c][0]*y0 + wv[c][1]*m0 + wv[c][2]*y1 + wv[c][3]*m1;
      D[c] = wd[c][0]*y0 + wd[c][1]*m0 + wd[c][2]*y1 + wd[c][3]*m1;
    }

    const KK_FLOAT x12 = X[1] * X[2];

    e3 += X[0] * x12;
    g0 += D[0] * x12;
    g1 += X[0] * D[1] * X[2];
    g2 += X[0] * X[1] * D[2];
  }

  if (eflag) energy += e3;

  KK_FLOAT fs[3];
  fs[canon_to_runtime[0]] = g0 / dx[canon_to_runtime[0]];
  fs[canon_to_runtime[1]] = g1 / dx[canon_to_runtime[1]];
  fs[canon_to_runtime[2]] = g2 / dx[canon_to_runtime[2]];

  // ij
  force[0] += fs[0] * dr[0];
  force[1] += fs[0] * dr[1];
  force[2] += fs[0] * dr[2];
  force[CHDIM+0] -= fs[0] * dr[0];
  force[CHDIM+1] -= fs[0] * dr[1];
  force[CHDIM+2] -= fs[0] * dr[2];

  // ik
  force[0] += fs[1] * dr[CHDIM+0];
  force[1] += fs[1] * dr[CHDIM+1];
  force[2] += fs[1] * dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1] * dr[CHDIM+0];
  force[2*CHDIM+1] -= fs[1] * dr[CHDIM+1];
  force[2*CHDIM+2] -= fs[1] * dr[CHDIM+2];

  // jk
  force[CHDIM+0] += fs[2] * dr[2*CHDIM+0];
  force[CHDIM+1] += fs[2] * dr[2*CHDIM+1];
  force[CHDIM+2] += fs[2] * dr[2*CHDIM+2];
  force[2*CHDIM+0] -= fs[2] * dr[2*CHDIM+0];
  force[2*CHDIM+1] -= fs[2] * dr[2*CHDIM+1];
  force[2*CHDIM+2] -= fs[2] * dr[2*CHDIM+2];

  if (vflag) {
    for (int p = 0; p < 3; p++) {
      stress[0] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+0];
      stress[1] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+1];
      stress[2] -= fs[p] * dr[p*CHDIM+0] * dr[p*CHDIM+2];
      stress[3] -= fs[p] * dr[p*CHDIM+1] * dr[p*CHDIM+1];
      stress[4] -= fs[p] * dr[p*CHDIM+1] * dr[p*CHDIM+2];
      stress[5] -= fs[p] * dr[p*CHDIM+2] * dr[p*CHDIM+2];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::interp_svd3x3_side_linear(
    int quadidx, bool left_side, const KK_FLOAT *rquery,
    KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1, KK_FLOAT *d2) const
{
  const int R = c_tab4b_svd3_rank[quadidx];
  const int n = c_tab4b_svd3_ngrid[quadidx];
  const int off = left_side ? c_tab4b_svd3_offset_left[quadidx]
                            : c_tab4b_svd3_offset_right[quadidx];

  KK_FLOAT r0x, r0y, r0z, invx, invy, invz, drx, dry, drz;
  if (left_side) {
    r0x = c_tab4b_svd3_r0_left(quadidx,0);
    r0y = c_tab4b_svd3_r0_left(quadidx,1);
    r0z = c_tab4b_svd3_r0_left(quadidx,2);
    drx = c_tab4b_svd3_dr_left(quadidx,0);
    dry = c_tab4b_svd3_dr_left(quadidx,1);
    drz = c_tab4b_svd3_dr_left(quadidx,2);
    invx = c_tab4b_svd3_invdr_left(quadidx,0);
    invy = c_tab4b_svd3_invdr_left(quadidx,1);
    invz = c_tab4b_svd3_invdr_left(quadidx,2);
  } else {
    r0x = c_tab4b_svd3_r0_right(quadidx,0);
    r0y = c_tab4b_svd3_r0_right(quadidx,1);
    r0z = c_tab4b_svd3_r0_right(quadidx,2);
    drx = c_tab4b_svd3_dr_right(quadidx,0);
    dry = c_tab4b_svd3_dr_right(quadidx,1);
    drz = c_tab4b_svd3_dr_right(quadidx,2);
    invx = c_tab4b_svd3_invdr_right(quadidx,0);
    invy = c_tab4b_svd3_invdr_right(quadidx,1);
    invz = c_tab4b_svd3_invdr_right(quadidx,2);
  }

  int i0 = (int)((rquery[0] - r0x) * invx);
  int i1 = (int)((rquery[1] - r0y) * invy);
  int i2 = (int)((rquery[2] - r0z) * invz);

  if (i0 < 0) i0 = 0; if (i0 > n-2) i0 = n-2;
  if (i1 < 0) i1 = 0; if (i1 > n-2) i1 = n-2;
  if (i2 < 0) i2 = 0; if (i2 > n-2) i2 = n-2;

  KK_FLOAT t0 = (rquery[0] - (r0x + i0*drx)) * invx;
  KK_FLOAT t1 = (rquery[1] - (r0y + i1*dry)) * invy;
  KK_FLOAT t2 = (rquery[2] - (r0z + i2*drz)) * invz;

  if (t0 < 0.0) t0 = 0.0; if (t0 > 1.0) t0 = 1.0;
  if (t1 < 0.0) t1 = 0.0; if (t1 > 1.0) t1 = 1.0;
  if (t2 < 0.0) t2 = 0.0; if (t2 > 1.0) t2 = 1.0;

  const KK_FLOAT a0 = 1.0 - t0, b0 = t0;
  const KK_FLOAT a1 = 1.0 - t1, b1 = t1;
  const KK_FLOAT a2 = 1.0 - t2, b2 = t2;

  const int s0 = n*n;
  const int s1 = n;

  const int idx000 = i0*s0 + i1*s1 + i2;
  const int idx001 = idx000 + 1;
  const int idx010 = idx000 + s1;
  const int idx011 = idx010 + 1;
  const int idx100 = idx000 + s0;
  const int idx101 = idx100 + 1;
  const int idx110 = idx100 + s1;
  const int idx111 = idx110 + 1;

  const KK_FLOAT w000 = a0*a1*a2;
  const KK_FLOAT w001 = a0*a1*b2;
  const KK_FLOAT w010 = a0*b1*a2;
  const KK_FLOAT w011 = a0*b1*b2;
  const KK_FLOAT w100 = b0*a1*a2;
  const KK_FLOAT w101 = b0*a1*b2;
  const KK_FLOAT w110 = b0*b1*a2;
  const KK_FLOAT w111 = b0*b1*b2;

  const float *V  = left_side ? d_tab4b_svd3_left_val.data()  : d_tab4b_svd3_right_val.data();
  const float *D0 = left_side ? d_tab4b_svd3_left_d0.data()   : d_tab4b_svd3_right_d0.data();
  const float *D1 = left_side ? d_tab4b_svd3_left_d1.data()   : d_tab4b_svd3_right_d1.data();
  const float *D2 = left_side ? d_tab4b_svd3_left_d2.data()   : d_tab4b_svd3_right_d2.data();

  const int o000 = off + idx000 * R;
  const int o001 = off + idx001 * R;
  const int o010 = off + idx010 * R;
  const int o011 = off + idx011 * R;
  const int o100 = off + idx100 * R;
  const int o101 = off + idx101 * R;
  const int o110 = off + idx110 * R;
  const int o111 = off + idx111 * R;

  for (int p = 0; p < R; p++) {
    val[p] =
      w000*(KK_FLOAT)V[o000+p] + w001*(KK_FLOAT)V[o001+p] +
      w010*(KK_FLOAT)V[o010+p] + w011*(KK_FLOAT)V[o011+p] +
      w100*(KK_FLOAT)V[o100+p] + w101*(KK_FLOAT)V[o101+p] +
      w110*(KK_FLOAT)V[o110+p] + w111*(KK_FLOAT)V[o111+p];

    d0[p] =
      w000*(KK_FLOAT)D0[o000+p] + w001*(KK_FLOAT)D0[o001+p] +
      w010*(KK_FLOAT)D0[o010+p] + w011*(KK_FLOAT)D0[o011+p] +
      w100*(KK_FLOAT)D0[o100+p] + w101*(KK_FLOAT)D0[o101+p] +
      w110*(KK_FLOAT)D0[o110+p] + w111*(KK_FLOAT)D0[o111+p];

    d1[p] =
      w000*(KK_FLOAT)D1[o000+p] + w001*(KK_FLOAT)D1[o001+p] +
      w010*(KK_FLOAT)D1[o010+p] + w011*(KK_FLOAT)D1[o011+p] +
      w100*(KK_FLOAT)D1[o100+p] + w101*(KK_FLOAT)D1[o101+p] +
      w110*(KK_FLOAT)D1[o110+p] + w111*(KK_FLOAT)D1[o111+p];

    d2[p] =
      w000*(KK_FLOAT)D2[o000+p] + w001*(KK_FLOAT)D2[o001+p] +
      w010*(KK_FLOAT)D2[o010+p] + w011*(KK_FLOAT)D2[o011+p] +
      w100*(KK_FLOAT)D2[o100+p] + w101*(KK_FLOAT)D2[o101+p] +
      w110*(KK_FLOAT)D2[o110+p] + w111*(KK_FLOAT)D2[o111+p];
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int chimesFFKokkos<DeviceType>::cp1d_weights(const KK_FLOAT r, const KK_FLOAT r0, const KK_FLOAT h, const KK_FLOAT invh, const int ngrid, const KK_FLOAT rc, KK_FLOAT* wv, KK_FLOAT* wd) const
{
  // Weights for one lookup in a CP 1D table.  With y0, m0 the value and
  // r-derivative of a rank term at the returned row i, and y1, m1 at row i+1:
  //
  //   X(r)  = wv[0]*y0 + wv[1]*m0 + wv[2]*y1 + wv[3]*m1
  //   X'(r) = wd[0]*y0 + wd[1]*m0 + wd[2]*y1 + wd[3]*m1
  //
  // The weights depend only on r, so a kernel computes them once per slot and
  // reuses them for every rank term.  Inside the grid this is cubic Hermite
  // interpolation (X' is the exact derivative of X).  Below the first grid
  // point (the inner cutoff) the function is continued exactly like
  // set_polys_out_of_range continues the polynomials, as the CPU
  // chimesFF::interpolateCP1DLinear does:
  //
  //   p(r) = p(r0) + s (exp((r-r0)/s) - 1) p'(r0),   s = inner_smooth_distance,
  //
  // applied to p = X / fcut.  For Tersoff cutoffs fcut = 1 at r0 and this is
  // X(r) = y0 + s (exp((r-r0)/s) - 1) m0.

  const KK_FLOAT s = (r - r0) * invh;

  if (s < 0.0) {
    const KK_FLOAT sm = inner_smooth_distance;
    const KK_FLOAT damp = exp((r - r0) / sm);
    const KK_FLOAT a = sm * (damp - 1.0);

    KK_FLOAT f0 = 1.0, fd0 = 0.0, f = 1.0, fd = 0.0;

    if (rc > r0) {
      get_fcut(r0, rc, f0, fd0);
      get_fcut(r, rc, f, fd);
    }

    if (f0 != 1.0 || fd0 != 0.0) {
      // p0 = y0/f0, p0' = (m0 - fd0*p0)/f0, p = p0 + a*p0', X = f*p, X' = fd*p + f*damp*p0'
      const KK_FLOAT inv_f0 = 1.0 / f0;
      const KK_FLOAT py = inv_f0 - a * fd0 * inv_f0 * inv_f0;
      const KK_FLOAT pm = a * inv_f0;
      const KK_FLOAT gy = -fd0 * inv_f0 * inv_f0;
      const KK_FLOAT gm = inv_f0;

      wv[0] = f * py;
      wv[1] = f * pm;
      wd[0] = fd * py + f * damp * gy;
      wd[1] = fd * pm + f * damp * gm;
    } else {
      wv[0] = 1.0;
      wv[1] = a;
      wd[0] = 0.0;
      wd[1] = damp;
    }

    wv[2] = 0.0;
    wv[3] = 0.0;
    wd[2] = 0.0;
    wd[3] = 0.0;

    return 0;
  }

  int i = (int)s;
  if (i > ngrid - 2) i = ngrid - 2;

  KK_FLOAT t = s - i;
  if (t > 1.0) t = 1.0;

  const KK_FLOAT t2 = t * t;
  const KK_FLOAT t3 = t2 * t;

  wv[0] =  2.0*t3 - 3.0*t2 + 1.0;
  wv[1] = (t3 - 2.0*t2 + t) * h;
  wv[2] = -2.0*t3 + 3.0*t2;
  wv[3] = (t3 - t2) * h;

  wd[0] = (6.0*t2 - 6.0*t) * invh;
  wd[1] =  3.0*t2 - 4.0*t + 1.0;
  wd[2] = -wd[0];
  wd[3] =  3.0*t2 - 2.0*t;

  return i;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B_cp_tab(
    const KK_FLOAT* dx,
    const KK_FLOAT* dr,
    const int* typ_idxs,
    KK_FLOAT* force,
    KK_FLOAT* stress,
    KK_FLOAT& energy) const
{
  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
                + typ_idxs[1]*natmtyps*natmtyps
                + typ_idxs[2]*natmtyps
                + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];
  if (quadidx < 0) return;

  const int R = c_tab4b_cp_rank[quadidx];
  if (R <= 0) {
    compute_4B(dx, dr, typ_idxs, force, stress, energy);
    return;
  }

  const int pmap0 = c_pair_int_quad_map(idx,0);
  const int pmap1 = c_pair_int_quad_map(idx,1);
  const int pmap2 = c_pair_int_quad_map(idx,2);
  const int pmap3 = c_pair_int_quad_map(idx,3);
  const int pmap4 = c_pair_int_quad_map(idx,4);
  const int pmap5 = c_pair_int_quad_map(idx,5);

  if (dx[0] >= c_chimes_4b_cutoff(quadidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_4b_cutoff(quadidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_4b_cutoff(quadidx,pmap2,1)) return;
  if (dx[3] >= c_chimes_4b_cutoff(quadidx,pmap3,1)) return;
  if (dx[4] >= c_chimes_4b_cutoff(quadidx,pmap4,1)) return;
  if (dx[5] >= c_chimes_4b_cutoff(quadidx,pmap5,1)) return;

  int param_to_runtime[6];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;
  param_to_runtime[pmap3] = 3;
  param_to_runtime[pmap4] = 4;
  param_to_runtime[pmap5] = 5;

  int canon_to_runtime[6];

  // Interpolation weights and the first of the two table rows for each
  // canonical slot (computed once, used for every rank term)

  const chimes_tab_real *V = d_tab4b_cp_vd.data();
  const chimes_tab_real *row[6];
  KK_FLOAT wv[6][4];
  KK_FLOAT wd[6][4];

  for (int c = 0; c < 6; c++) {
    const int pslot = c_tab4b_cp_canon_to_param(quadidx,c);
    canon_to_runtime[c] = param_to_runtime[pslot];

    const int i = cp1d_weights(dx[canon_to_runtime[c]], c_tab4b_cp_r0(quadidx,c), c_tab4b_cp_dr(quadidx,c), c_tab4b_cp_invdr(quadidx,c), c_tab4b_cp_ngrid(quadidx,c), c_chimes_4b_cutoff(quadidx,pslot,1), wv[c], wd[c]);

    row[c] = V + c_tab4b_cp_offset(quadidx,c) + (size_t)i * (size_t)(2 * R);
  }

  KK_FLOAT e4 = 0.0;

  KK_FLOAT dE_canon[6];
  for (int c = 0; c < 6; c++) dE_canon[c] = 0.0;

  for (int q = 0; q < R; q++) {
    KK_FLOAT X[6];
    KK_FLOAT D[6];

    for (int c = 0; c < 6; c++) {
      const chimes_tab_real *p = row[c];

      const KK_FLOAT y0 = p[q];
      const KK_FLOAT m0 = p[R + q];
      const KK_FLOAT y1 = p[2*R + q];
      const KK_FLOAT m1 = p[3*R + q];

      X[c] = wv[c][0]*y0 + wv[c][1]*m0 + wv[c][2]*y1 + wv[c][3]*m1;
      D[c] = wd[c][0]*y0 + wd[c][1]*m0 + wd[c][2]*y1 + wd[c][3]*m1;
    }

    const KK_FLOAT x01 = X[0] * X[1];
    const KK_FLOAT x23 = X[2] * X[3];
    const KK_FLOAT x45 = X[4] * X[5];

    const KK_FLOAT prod = x01 * x23 * x45;

    e4 += prod;

    dE_canon[0] += D[0] * X[1] * x23 * x45;
    dE_canon[1] += X[0] * D[1] * x23 * x45;
    dE_canon[2] += x01 * D[2] * X[3] * x45;
    dE_canon[3] += x01 * X[2] * D[3] * x45;
    dE_canon[4] += x01 * x23 * D[4] * X[5];
    dE_canon[5] += x01 * x23 * X[4] * D[5];
  }

  if (eflag) energy += e4;

  KK_FLOAT fs[6];

  for (int p = 0; p < 6; p++)
    fs[p] = 0.0;

  for (int c = 0; c < 6; c++) {
    const int rslot = canon_to_runtime[c];
    fs[rslot] = dE_canon[c] / dx[rslot];
  }

  // Forces
  force[0] += fs[0]*dr[0]; force[1] += fs[0]*dr[1]; force[2] += fs[0]*dr[2];
  force[CHDIM+0] -= fs[0]*dr[0]; force[CHDIM+1] -= fs[0]*dr[1]; force[CHDIM+2] -= fs[0]*dr[2];

  force[0] += fs[1]*dr[CHDIM+0]; force[1] += fs[1]*dr[CHDIM+1]; force[2] += fs[1]*dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1]*dr[CHDIM+0]; force[2*CHDIM+1] -= fs[1]*dr[CHDIM+1]; force[2*CHDIM+2] -= fs[1]*dr[CHDIM+2];

  force[0] += fs[2]*dr[2*CHDIM+0]; force[1] += fs[2]*dr[2*CHDIM+1]; force[2] += fs[2]*dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fs[2]*dr[2*CHDIM+0]; force[3*CHDIM+1] -= fs[2]*dr[2*CHDIM+1]; force[3*CHDIM+2] -= fs[2]*dr[2*CHDIM+2];

  force[CHDIM+0] += fs[3]*dr[3*CHDIM+0]; force[CHDIM+1] += fs[3]*dr[3*CHDIM+1]; force[CHDIM+2] += fs[3]*dr[3*CHDIM+2];
  force[2*CHDIM+0] -= fs[3]*dr[3*CHDIM+0]; force[2*CHDIM+1] -= fs[3]*dr[3*CHDIM+1]; force[2*CHDIM+2] -= fs[3]*dr[3*CHDIM+2];

  force[CHDIM+0] += fs[4]*dr[4*CHDIM+0]; force[CHDIM+1] += fs[4]*dr[4*CHDIM+1]; force[CHDIM+2] += fs[4]*dr[4*CHDIM+2];
  force[3*CHDIM+0] -= fs[4]*dr[4*CHDIM+0]; force[3*CHDIM+1] -= fs[4]*dr[4*CHDIM+1]; force[3*CHDIM+2] -= fs[4]*dr[4*CHDIM+2];

  force[2*CHDIM+0] += fs[5]*dr[5*CHDIM+0]; force[2*CHDIM+1] += fs[5]*dr[5*CHDIM+1]; force[2*CHDIM+2] += fs[5]*dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fs[5]*dr[5*CHDIM+0]; force[3*CHDIM+1] -= fs[5]*dr[5*CHDIM+1]; force[3*CHDIM+2] -= fs[5]*dr[5*CHDIM+2];

  if (vflag) {
    for (int p = 0; p < 6; p++) {
      stress[0] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+0];
      stress[1] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+1];
      stress[2] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+2];
      stress[3] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+1];
      stress[4] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+2];
      stress[5] -= fs[p]*dr[p*CHDIM+2]*dr[p*CHDIM+2];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B_cp_direct(
    const KK_FLOAT* dx,
    const KK_FLOAT* dr,
    const int* typ_idxs,
    KK_FLOAT* force,
    KK_FLOAT* stress,
    KK_FLOAT& energy) const
{
  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
                + typ_idxs[1]*natmtyps*natmtyps
                + typ_idxs[2]*natmtyps
                + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];
  if (quadidx < 0) return;

  const int R = c_tab4b_cp_direct_rank[quadidx];
  if (R <= 0) {
    compute_4B(dx, dr, typ_idxs, force, stress, energy);
    return;
  }

  const int pmap0 = c_pair_int_quad_map(idx,0);
  const int pmap1 = c_pair_int_quad_map(idx,1);
  const int pmap2 = c_pair_int_quad_map(idx,2);
  const int pmap3 = c_pair_int_quad_map(idx,3);
  const int pmap4 = c_pair_int_quad_map(idx,4);
  const int pmap5 = c_pair_int_quad_map(idx,5);

  if (dx[0] >= c_chimes_4b_cutoff(quadidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_4b_cutoff(quadidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_4b_cutoff(quadidx,pmap2,1)) return;
  if (dx[3] >= c_chimes_4b_cutoff(quadidx,pmap3,1)) return;
  if (dx[4] >= c_chimes_4b_cutoff(quadidx,pmap4,1)) return;
  if (dx[5] >= c_chimes_4b_cutoff(quadidx,pmap5,1)) return;

  int param_to_runtime[6];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;
  param_to_runtime[pmap3] = 3;
  param_to_runtime[pmap4] = 4;
  param_to_runtime[pmap5] = 5;

  int canon_to_runtime[6];

  for (int c = 0; c < 6; c++)
    canon_to_runtime[c] =
        param_to_runtime[c_tab4b_cp_direct_canon_to_param(quadidx,c)];

  int pair_type_runtime[6];

  pair_type_runtime[0] = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
  pair_type_runtime[1] = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
  pair_type_runtime[2] = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[3]);
  pair_type_runtime[3] = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
  pair_type_runtime[4] = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[3]);
  pair_type_runtime[5] = c_atom_int_pair_map(typ_idxs[2]*natmtyps + typ_idxs[3]);

  KK_FLOAT G[6][MAX_4B_POLY];
  KK_FLOAT DG[6][MAX_4B_POLY];

  for (int c = 0; c < 6; c++) {
    const int rslot = canon_to_runtime[c];
    const int pslot = c_pair_int_quad_map(idx,rslot);
    const int pair_type = pair_type_runtime[rslot];

    const int order = c_tab4b_cp_direct_orders(quadidx,c);

    KK_FLOAT Tn[MAX_4B_POLY];
    KK_FLOAT Tnd[MAX_4B_POLY];

    const KK_FLOAT rmin = c_chimes_4b_cutoff(quadidx,pslot,0);
    const KK_FLOAT rmax = c_chimes_4b_cutoff(quadidx,pslot,1);

    set_cheby_polys(
        Tn,
        Tnd,
        dx[rslot],
        c_morse_var[pair_type],
        rmin,
        rmax,
        order);

    KK_FLOAT fcut;
    KK_FLOAT fcutderiv;

    get_fcut(dx[rslot], rmax, fcut, fcutderiv);

    for (int p = 0; p <= order; p++) {
      G[c][p] = fcut * Tn[p];
      DG[c][p] = fcutderiv * Tn[p] + fcut * Tnd[p];
    }
  }

  KK_FLOAT e4 = 0.0;

  KK_FLOAT dE_canon[6];
  for (int c = 0; c < 6; c++)
    dE_canon[c] = 0.0;

  int slot_order[6];
  int slot_off[6];

  for (int c = 0; c < 6; c++) {
    slot_order[c] = c_tab4b_cp_direct_orders(quadidx,c);
    slot_off[c] = c_tab4b_cp_direct_offset(quadidx,c);
  }

  for (int q = 0; q < R; q++) {
    KK_FLOAT X[6];
    KK_FLOAT D[6];

    for (int c = 0; c < 6; c++) {
      const int order = slot_order[c];
      const int off = slot_off[c];

      KK_FLOAT x = 0.0;
      KK_FLOAT d = 0.0;

      for (int p = 0; p <= order; p++) {
        const KK_FLOAT A = c_tab4b_cp_direct_factor[off + p*R + q];

        x += A * G[c][p];
        d += A * DG[c][p];
      }

      X[c] = x;
      D[c] = d;
    }

    const KK_FLOAT x01 = X[0] * X[1];
    const KK_FLOAT x23 = X[2] * X[3];
    const KK_FLOAT x45 = X[4] * X[5];

    const KK_FLOAT prod = x01 * x23 * x45;

    e4 += prod;

    dE_canon[0] += D[0] * X[1] * x23 * x45;
    dE_canon[1] += X[0] * D[1] * x23 * x45;
    dE_canon[2] += x01 * D[2] * X[3] * x45;
    dE_canon[3] += x01 * X[2] * D[3] * x45;
    dE_canon[4] += x01 * x23 * D[4] * X[5];
    dE_canon[5] += x01 * x23 * X[4] * D[5];
  }

  if (eflag) energy += e4;

  KK_FLOAT fs[6];

  for (int p = 0; p < 6; p++)
    fs[p] = 0.0;

  for (int c = 0; c < 6; c++) {
    const int rslot = canon_to_runtime[c];
    fs[rslot] = dE_canon[c] / dx[rslot];
  }

  force[0] += fs[0]*dr[0]; force[1] += fs[0]*dr[1]; force[2] += fs[0]*dr[2];
  force[CHDIM+0] -= fs[0]*dr[0]; force[CHDIM+1] -= fs[0]*dr[1]; force[CHDIM+2] -= fs[0]*dr[2];

  force[0] += fs[1]*dr[CHDIM+0]; force[1] += fs[1]*dr[CHDIM+1]; force[2] += fs[1]*dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1]*dr[CHDIM+0]; force[2*CHDIM+1] -= fs[1]*dr[CHDIM+1]; force[2*CHDIM+2] -= fs[1]*dr[CHDIM+2];

  force[0] += fs[2]*dr[2*CHDIM+0]; force[1] += fs[2]*dr[2*CHDIM+1]; force[2] += fs[2]*dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fs[2]*dr[2*CHDIM+0]; force[3*CHDIM+1] -= fs[2]*dr[2*CHDIM+1]; force[3*CHDIM+2] -= fs[2]*dr[2*CHDIM+2];

  force[CHDIM+0] += fs[3]*dr[3*CHDIM+0]; force[CHDIM+1] += fs[3]*dr[3*CHDIM+1]; force[CHDIM+2] += fs[3]*dr[3*CHDIM+2];
  force[2*CHDIM+0] -= fs[3]*dr[3*CHDIM+0]; force[2*CHDIM+1] -= fs[3]*dr[3*CHDIM+1]; force[2*CHDIM+2] -= fs[3]*dr[3*CHDIM+2];

  force[CHDIM+0] += fs[4]*dr[4*CHDIM+0]; force[CHDIM+1] += fs[4]*dr[4*CHDIM+1]; force[CHDIM+2] += fs[4]*dr[4*CHDIM+2];
  force[3*CHDIM+0] -= fs[4]*dr[4*CHDIM+0]; force[3*CHDIM+1] -= fs[4]*dr[4*CHDIM+1]; force[3*CHDIM+2] -= fs[4]*dr[4*CHDIM+2];

  force[2*CHDIM+0] += fs[5]*dr[5*CHDIM+0]; force[2*CHDIM+1] += fs[5]*dr[5*CHDIM+1]; force[2*CHDIM+2] += fs[5]*dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fs[5]*dr[5*CHDIM+0]; force[3*CHDIM+1] -= fs[5]*dr[5*CHDIM+1]; force[3*CHDIM+2] -= fs[5]*dr[5*CHDIM+2];

  if (vflag) {
    for (int p = 0; p < 6; p++) {
      stress[0] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+0];
      stress[1] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+1];
      stress[2] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+2];
      stress[3] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+1];
      stress[4] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+2];
      stress[5] -= fs[p]*dr[p*CHDIM+2]*dr[p*CHDIM+2];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B_svd3x3_tab(
    const KK_FLOAT* dx,
    const KK_FLOAT* dr,
    const int* typ_idxs,
    KK_FLOAT* force,
    KK_FLOAT* stress,
    KK_FLOAT& energy) const
{
  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
                + typ_idxs[1]*natmtyps*natmtyps
                + typ_idxs[2]*natmtyps
                + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];
  if (quadidx < 0) return;

  const int R = c_tab4b_svd3_rank[quadidx];
  if (R <= 0) {
    compute_4B(dx, dr, typ_idxs, force, stress, energy);
    return;
  }

  const int pmap0 = c_pair_int_quad_map(idx,0);
  const int pmap1 = c_pair_int_quad_map(idx,1);
  const int pmap2 = c_pair_int_quad_map(idx,2);
  const int pmap3 = c_pair_int_quad_map(idx,3);
  const int pmap4 = c_pair_int_quad_map(idx,4);
  const int pmap5 = c_pair_int_quad_map(idx,5);

  if (dx[0] >= c_chimes_4b_cutoff(quadidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_4b_cutoff(quadidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_4b_cutoff(quadidx,pmap2,1)) return;
  if (dx[3] >= c_chimes_4b_cutoff(quadidx,pmap3,1)) return;
  if (dx[4] >= c_chimes_4b_cutoff(quadidx,pmap4,1)) return;
  if (dx[5] >= c_chimes_4b_cutoff(quadidx,pmap5,1)) return;

  int param_to_runtime[6];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;
  param_to_runtime[pmap3] = 3;
  param_to_runtime[pmap4] = 4;
  param_to_runtime[pmap5] = 5;

  int canon_to_runtime[6];
  for (int c=0; c<6; c++)
    canon_to_runtime[c] =
        param_to_runtime[c_tab4b_svd3_canon_to_param(quadidx,c)];

  KK_FLOAT rl[3], rr[3];

  rl[0] = dx[canon_to_runtime[0]];
  rl[1] = dx[canon_to_runtime[1]];
  rl[2] = dx[canon_to_runtime[2]];

  rr[0] = dx[canon_to_runtime[3]];
  rr[1] = dx[canon_to_runtime[4]];
  rr[2] = dx[canon_to_runtime[5]];

  const int n = c_tab4b_svd3_ngrid[quadidx];
  const int offL = c_tab4b_svd3_offset_left[quadidx];
  const int offR = c_tab4b_svd3_offset_right[quadidx];

  int iL[3], iR[3];
  KK_FLOAT tL[3], tR[3];

  for (int d=0; d<3; d++) {
    const KK_FLOAT r0L = c_tab4b_svd3_r0_left(quadidx,d);
    const KK_FLOAT drL = c_tab4b_svd3_dr_left(quadidx,d);
    const KK_FLOAT invL = c_tab4b_svd3_invdr_left(quadidx,d);

    iL[d] = (int)((rl[d] - r0L) * invL);
    if (iL[d] < 0) iL[d] = 0;
    if (iL[d] > n-2) iL[d] = n-2;

    tL[d] = (rl[d] - (r0L + iL[d]*drL)) * invL;
    if (tL[d] < 0.0) tL[d] = 0.0;
    if (tL[d] > 1.0) tL[d] = 1.0;

    const KK_FLOAT r0R = c_tab4b_svd3_r0_right(quadidx,d);
    const KK_FLOAT drR = c_tab4b_svd3_dr_right(quadidx,d);
    const KK_FLOAT invR = c_tab4b_svd3_invdr_right(quadidx,d);

    iR[d] = (int)((rr[d] - r0R) * invR);
    if (iR[d] < 0) iR[d] = 0;
    if (iR[d] > n-2) iR[d] = n-2;

    tR[d] = (rr[d] - (r0R + iR[d]*drR)) * invR;
    if (tR[d] < 0.0) tR[d] = 0.0;
    if (tR[d] > 1.0) tR[d] = 1.0;
  }

  const int rowL = iL[0]*n*n + iL[1]*n + iL[2];
  const int rowR = iR[0]*n*n + iR[1]*n + iR[2];

  const float *LV  = d_tab4b_svd3_left_val.data();
  const float *LD0 = d_tab4b_svd3_left_d0.data();
  const float *LD1 = d_tab4b_svd3_left_d1.data();
  const float *LD2 = d_tab4b_svd3_left_d2.data();

  const float *RV  = d_tab4b_svd3_right_val.data();
  const float *RD0 = d_tab4b_svd3_right_d0.data();
  const float *RD1 = d_tab4b_svd3_right_d1.data();
  const float *RD2 = d_tab4b_svd3_right_d2.data();

  KK_FLOAT e4 = 0.0;
  KK_FLOAT dL0 = 0.0, dL1 = 0.0, dL2 = 0.0;
  KK_FLOAT dR0 = 0.0, dR1 = 0.0, dR2 = 0.0;

  for (int p=0; p<R; p++) {
    KK_FLOAT X=0.0, X0=0.0, X1=0.0, X2=0.0;
    KK_FLOAT Y=0.0, Y0=0.0, Y1=0.0, Y2=0.0;

    for (int c=0; c<8; c++) {
      const int b0 = (c >> 0) & 1;
      const int b1 = (c >> 1) & 1;
      const int b2 = (c >> 2) & 1;

      const KK_FLOAT wL =
          (b0 ? tL[0] : 1.0 - tL[0]) *
          (b1 ? tL[1] : 1.0 - tL[1]) *
          (b2 ? tL[2] : 1.0 - tL[2]);

      const KK_FLOAT wR =
          (b0 ? tR[0] : 1.0 - tR[0]) *
          (b1 ? tR[1] : 1.0 - tR[1]) *
          (b2 ? tR[2] : 1.0 - tR[2]);

      const size_t oL = (size_t)offL + ((size_t)rowL*8u + (size_t)c)*(size_t)R + (size_t)p;
      const size_t oR = (size_t)offR + ((size_t)rowR*8u + (size_t)c)*(size_t)R + (size_t)p;

      X  += wL * (KK_FLOAT)LV[oL];
      X0 += wL * (KK_FLOAT)LD0[oL];
      X1 += wL * (KK_FLOAT)LD1[oL];
      X2 += wL * (KK_FLOAT)LD2[oL];

      Y  += wR * (KK_FLOAT)RV[oR];
      Y0 += wR * (KK_FLOAT)RD0[oR];
      Y1 += wR * (KK_FLOAT)RD1[oR];
      Y2 += wR * (KK_FLOAT)RD2[oR];
    }

    e4  += X * Y;
    dL0 += X0 * Y;
    dL1 += X1 * Y;
    dL2 += X2 * Y;
    dR0 += X * Y0;
    dR1 += X * Y1;
    dR2 += X * Y2;
  }

  if (eflag) energy += e4;

  KK_FLOAT fs[6];
  fs[canon_to_runtime[0]] = dL0 / dx[canon_to_runtime[0]];
  fs[canon_to_runtime[1]] = dL1 / dx[canon_to_runtime[1]];
  fs[canon_to_runtime[2]] = dL2 / dx[canon_to_runtime[2]];
  fs[canon_to_runtime[3]] = dR0 / dx[canon_to_runtime[3]];
  fs[canon_to_runtime[4]] = dR1 / dx[canon_to_runtime[4]];
  fs[canon_to_runtime[5]] = dR2 / dx[canon_to_runtime[5]];

  // ij
  force[0] += fs[0]*dr[0]; force[1] += fs[0]*dr[1]; force[2] += fs[0]*dr[2];
  force[CHDIM+0] -= fs[0]*dr[0]; force[CHDIM+1] -= fs[0]*dr[1]; force[CHDIM+2] -= fs[0]*dr[2];

  // ik
  force[0] += fs[1]*dr[CHDIM+0]; force[1] += fs[1]*dr[CHDIM+1]; force[2] += fs[1]*dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1]*dr[CHDIM+0]; force[2*CHDIM+1] -= fs[1]*dr[CHDIM+1]; force[2*CHDIM+2] -= fs[1]*dr[CHDIM+2];

  // il
  force[0] += fs[2]*dr[2*CHDIM+0]; force[1] += fs[2]*dr[2*CHDIM+1]; force[2] += fs[2]*dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fs[2]*dr[2*CHDIM+0]; force[3*CHDIM+1] -= fs[2]*dr[2*CHDIM+1]; force[3*CHDIM+2] -= fs[2]*dr[2*CHDIM+2];

  // jk
  force[CHDIM+0] += fs[3]*dr[3*CHDIM+0]; force[CHDIM+1] += fs[3]*dr[3*CHDIM+1]; force[CHDIM+2] += fs[3]*dr[3*CHDIM+2];
  force[2*CHDIM+0] -= fs[3]*dr[3*CHDIM+0]; force[2*CHDIM+1] -= fs[3]*dr[3*CHDIM+1]; force[2*CHDIM+2] -= fs[3]*dr[3*CHDIM+2];

  // jl
  force[CHDIM+0] += fs[4]*dr[4*CHDIM+0]; force[CHDIM+1] += fs[4]*dr[4*CHDIM+1]; force[CHDIM+2] += fs[4]*dr[4*CHDIM+2];
  force[3*CHDIM+0] -= fs[4]*dr[4*CHDIM+0]; force[3*CHDIM+1] -= fs[4]*dr[4*CHDIM+1]; force[3*CHDIM+2] -= fs[4]*dr[4*CHDIM+2];

  // kl
  force[2*CHDIM+0] += fs[5]*dr[5*CHDIM+0]; force[2*CHDIM+1] += fs[5]*dr[5*CHDIM+1]; force[2*CHDIM+2] += fs[5]*dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fs[5]*dr[5*CHDIM+0]; force[3*CHDIM+1] -= fs[5]*dr[5*CHDIM+1]; force[3*CHDIM+2] -= fs[5]*dr[5*CHDIM+2];

  if (vflag) {
    for (int p=0; p<6; p++) {
      stress[0] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+0];
      stress[1] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+1];
      stress[2] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+2];
      stress[3] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+1];
      stress[4] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+2];
      stress[5] -= fs[p]*dr[p*CHDIM+2]*dr[p*CHDIM+2];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::interp_svd4x2_left_linear(
    int quadidx, const KK_FLOAT *rquery,
    KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1, KK_FLOAT *d2, KK_FLOAT *d3) const
{
  const int R = c_tab4b_svd42_rank[quadidx];
  const int n = c_tab4b_svd42_ngrid_left[quadidx];
  const int off = c_tab4b_svd42_offset_left[quadidx];

  int i[4];
  KK_FLOAT t[4];
  for (int d = 0; d < 4; d++) {
    const KK_FLOAT r0 = c_tab4b_svd42_r0_left(quadidx,d);
    const KK_FLOAT dr = c_tab4b_svd42_dr_left(quadidx,d);
    const KK_FLOAT invdr = c_tab4b_svd42_invdr_left(quadidx,d);
    i[d] = (int)((rquery[d] - r0) * invdr);
    if (i[d] < 0) i[d] = 0;
    if (i[d] > n-2) i[d] = n-2;
    t[d] = (rquery[d] - (r0 + i[d]*dr)) * invdr;
    if (t[d] < 0.0) t[d] = 0.0;
    if (t[d] > 1.0) t[d] = 1.0;
  }

  for (int p = 0; p < R; p++) {
    val[p] = 0.0; d0[p] = 0.0; d1[p] = 0.0; d2[p] = 0.0; d3[p] = 0.0;
  }

  const int s0 = n*n*n;
  const int s1 = n*n;
  const int s2 = n;

  const float *V  = d_tab4b_svd42_left_val.data();
  const float *D0 = d_tab4b_svd42_left_d0.data();
  const float *D1 = d_tab4b_svd42_left_d1.data();
  const float *D2 = d_tab4b_svd42_left_d2.data();
  const float *D3 = d_tab4b_svd42_left_d3.data();

  for (int corner = 0; corner < 16; corner++) {
    KK_FLOAT w = 1.0;
    int idx = 0;

    int b0 = (corner >> 0) & 1;
    int b1 = (corner >> 1) & 1;
    int b2 = (corner >> 2) & 1;
    int b3 = (corner >> 3) & 1;

    int j0 = i[0] + b0;
    int j1 = i[1] + b1;
    int j2 = i[2] + b2;
    int j3 = i[3] + b3;

    w *= b0 ? t[0] : (1.0 - t[0]);
    w *= b1 ? t[1] : (1.0 - t[1]);
    w *= b2 ? t[2] : (1.0 - t[2]);
    w *= b3 ? t[3] : (1.0 - t[3]);

    idx = j0*s0 + j1*s1 + j2*s2 + j3;
    const int o = off + idx * R;

    for (int p = 0; p < R; p++) {
      val[p] += w * (KK_FLOAT)V[o+p];
      d0[p]  += w * (KK_FLOAT)D0[o+p];
      d1[p]  += w * (KK_FLOAT)D1[o+p];
      d2[p]  += w * (KK_FLOAT)D2[o+p];
      d3[p]  += w * (KK_FLOAT)D3[o+p];
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::interp_svd4x2_right_linear(
    int quadidx, const KK_FLOAT *rquery,
    KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1) const
{
  const int R = c_tab4b_svd42_rank[quadidx];
  const int n = c_tab4b_svd42_ngrid_right[quadidx];
  const int off = c_tab4b_svd42_offset_right[quadidx];

  const KK_FLOAT r00 = c_tab4b_svd42_r0_right(quadidx,0);
  const KK_FLOAT r01 = c_tab4b_svd42_r0_right(quadidx,1);
  const KK_FLOAT dr0 = c_tab4b_svd42_dr_right(quadidx,0);
  const KK_FLOAT dr1 = c_tab4b_svd42_dr_right(quadidx,1);
  const KK_FLOAT inv0 = c_tab4b_svd42_invdr_right(quadidx,0);
  const KK_FLOAT inv1 = c_tab4b_svd42_invdr_right(quadidx,1);

  int i0 = (int)((rquery[0] - r00) * inv0);
  int i1 = (int)((rquery[1] - r01) * inv1);
  if (i0 < 0) i0 = 0; if (i0 > n-2) i0 = n-2;
  if (i1 < 0) i1 = 0; if (i1 > n-2) i1 = n-2;

  KK_FLOAT t0 = (rquery[0] - (r00 + i0*dr0)) * inv0;
  KK_FLOAT t1 = (rquery[1] - (r01 + i1*dr1)) * inv1;
  if (t0 < 0.0) t0 = 0.0; if (t0 > 1.0) t0 = 1.0;
  if (t1 < 0.0) t1 = 0.0; if (t1 > 1.0) t1 = 1.0;

  const KK_FLOAT w00 = (1.0-t0)*(1.0-t1);
  const KK_FLOAT w01 = (1.0-t0)*t1;
  const KK_FLOAT w10 = t0*(1.0-t1);
  const KK_FLOAT w11 = t0*t1;

  const int idx00 = i0*n + i1;
  const int idx01 = idx00 + 1;
  const int idx10 = idx00 + n;
  const int idx11 = idx10 + 1;

  const float *V  = d_tab4b_svd42_right_val.data();
  const float *D0 = d_tab4b_svd42_right_d0.data();
  const float *D1 = d_tab4b_svd42_right_d1.data();

  const int o00 = off + idx00 * R;
  const int o01 = off + idx01 * R;
  const int o10 = off + idx10 * R;
  const int o11 = off + idx11 * R;

  for (int p = 0; p < R; p++) {
    val[p] =
      w00*(KK_FLOAT)V[o00+p] + w01*(KK_FLOAT)V[o01+p] +
      w10*(KK_FLOAT)V[o10+p] + w11*(KK_FLOAT)V[o11+p];
    d0[p] =
      w00*(KK_FLOAT)D0[o00+p] + w01*(KK_FLOAT)D0[o01+p] +
      w10*(KK_FLOAT)D0[o10+p] + w11*(KK_FLOAT)D0[o11+p];
    d1[p] =
      w00*(KK_FLOAT)D1[o00+p] + w01*(KK_FLOAT)D1[o01+p] +
      w10*(KK_FLOAT)D1[o10+p] + w11*(KK_FLOAT)D1[o11+p];
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B_svd4x2_tab(
    const KK_FLOAT* dx,
    const KK_FLOAT* dr,
    const int* typ_idxs,
    KK_FLOAT* force,
    KK_FLOAT* stress,
    KK_FLOAT& energy) const
{
  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
                + typ_idxs[1]*natmtyps*natmtyps
                + typ_idxs[2]*natmtyps
                + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];
  if (quadidx < 0) return;

  const int R = c_tab4b_svd42_rank[quadidx];
  if (R <= 0) {
    compute_4B(dx, dr, typ_idxs, force, stress, energy);
    return;
  }

  const int pmap0 = c_pair_int_quad_map(idx,0);
  const int pmap1 = c_pair_int_quad_map(idx,1);
  const int pmap2 = c_pair_int_quad_map(idx,2);
  const int pmap3 = c_pair_int_quad_map(idx,3);
  const int pmap4 = c_pair_int_quad_map(idx,4);
  const int pmap5 = c_pair_int_quad_map(idx,5);

  if (dx[0] >= c_chimes_4b_cutoff(quadidx,pmap0,1)) return;
  if (dx[1] >= c_chimes_4b_cutoff(quadidx,pmap1,1)) return;
  if (dx[2] >= c_chimes_4b_cutoff(quadidx,pmap2,1)) return;
  if (dx[3] >= c_chimes_4b_cutoff(quadidx,pmap3,1)) return;
  if (dx[4] >= c_chimes_4b_cutoff(quadidx,pmap4,1)) return;
  if (dx[5] >= c_chimes_4b_cutoff(quadidx,pmap5,1)) return;

  int param_to_runtime[6];
  param_to_runtime[pmap0] = 0;
  param_to_runtime[pmap1] = 1;
  param_to_runtime[pmap2] = 2;
  param_to_runtime[pmap3] = 3;
  param_to_runtime[pmap4] = 4;
  param_to_runtime[pmap5] = 5;

  int canon_to_runtime[6];
  for (int c=0; c<6; c++)
    canon_to_runtime[c] =
        param_to_runtime[c_tab4b_svd42_canon_to_param(quadidx,c)];

  KK_FLOAT rl[4], rr[2];

  rl[0] = dx[canon_to_runtime[0]];
  rl[1] = dx[canon_to_runtime[1]];
  rl[2] = dx[canon_to_runtime[2]];
  rl[3] = dx[canon_to_runtime[3]];

  rr[0] = dx[canon_to_runtime[4]];
  rr[1] = dx[canon_to_runtime[5]];

  const int nL = c_tab4b_svd42_ngrid_left[quadidx];
  const int nR = c_tab4b_svd42_ngrid_right[quadidx];

  const int offL = c_tab4b_svd42_offset_left[quadidx];
  const int offR = c_tab4b_svd42_offset_right[quadidx];

  int iL[4];
  KK_FLOAT tL[4];

  for (int d=0; d<4; d++) {
    const KK_FLOAT r0 = c_tab4b_svd42_r0_left(quadidx,d);
    const KK_FLOAT dg = c_tab4b_svd42_dr_left(quadidx,d);
    const KK_FLOAT iv = c_tab4b_svd42_invdr_left(quadidx,d);

    iL[d] = (int)((rl[d] - r0) * iv);
    if (iL[d] < 0) iL[d] = 0;
    if (iL[d] > nL-2) iL[d] = nL-2;

    tL[d] = (rl[d] - (r0 + iL[d]*dg)) * iv;
    if (tL[d] < 0.0) tL[d] = 0.0;
    if (tL[d] > 1.0) tL[d] = 1.0;
  }

  int iR[2];
  KK_FLOAT tR[2];

  for (int d=0; d<2; d++) {
    const KK_FLOAT r0 = c_tab4b_svd42_r0_right(quadidx,d);
    const KK_FLOAT dg = c_tab4b_svd42_dr_right(quadidx,d);
    const KK_FLOAT iv = c_tab4b_svd42_invdr_right(quadidx,d);

    iR[d] = (int)((rr[d] - r0) * iv);
    if (iR[d] < 0) iR[d] = 0;
    if (iR[d] > nR-2) iR[d] = nR-2;

    tR[d] = (rr[d] - (r0 + iR[d]*dg)) * iv;
    if (tR[d] < 0.0) tR[d] = 0.0;
    if (tR[d] > 1.0) tR[d] = 1.0;
  }

  const int rowL =
      iL[0]*nL*nL*nL + iL[1]*nL*nL + iL[2]*nL + iL[3];

  const int rowR =
      iR[0]*nR + iR[1];

  const float *LV  = d_tab4b_svd42_left_val.data();
  const float *LD0 = d_tab4b_svd42_left_d0.data();
  const float *LD1 = d_tab4b_svd42_left_d1.data();
  const float *LD2 = d_tab4b_svd42_left_d2.data();
  const float *LD3 = d_tab4b_svd42_left_d3.data();

  const float *RV  = d_tab4b_svd42_right_val.data();
  const float *RD0 = d_tab4b_svd42_right_d0.data();
  const float *RD1 = d_tab4b_svd42_right_d1.data();

  KK_FLOAT e4 = 0.0;
  KK_FLOAT dL0 = 0.0, dL1 = 0.0, dL2 = 0.0, dL3 = 0.0;
  KK_FLOAT dR0 = 0.0, dR1 = 0.0;

  for (int p=0; p<R; p++) {
    KK_FLOAT L=0.0, L0=0.0, L1=0.0, L2=0.0, L3=0.0;
    KK_FLOAT Y=0.0, Y0=0.0, Y1=0.0;

    for (int c=0; c<16; c++) {
      const int b0 = (c >> 0) & 1;
      const int b1 = (c >> 1) & 1;
      const int b2 = (c >> 2) & 1;
      const int b3 = (c >> 3) & 1;

      const KK_FLOAT w =
          (b0 ? tL[0] : 1.0 - tL[0]) *
          (b1 ? tL[1] : 1.0 - tL[1]) *
          (b2 ? tL[2] : 1.0 - tL[2]) *
          (b3 ? tL[3] : 1.0 - tL[3]);

      const size_t o =
          (size_t)offL + ((size_t)rowL*16u + (size_t)c)*(size_t)R + (size_t)p;

      L  += w * (KK_FLOAT)LV[o];
      L0 += w * (KK_FLOAT)LD0[o];
      L1 += w * (KK_FLOAT)LD1[o];
      L2 += w * (KK_FLOAT)LD2[o];
      L3 += w * (KK_FLOAT)LD3[o];
    }

    for (int c=0; c<4; c++) {
      const int b0 = (c >> 0) & 1;
      const int b1 = (c >> 1) & 1;

      const KK_FLOAT w =
          (b0 ? tR[0] : 1.0 - tR[0]) *
          (b1 ? tR[1] : 1.0 - tR[1]);

      const size_t o =
          (size_t)offR + ((size_t)rowR*4u + (size_t)c)*(size_t)R + (size_t)p;

      Y  += w * (KK_FLOAT)RV[o];
      Y0 += w * (KK_FLOAT)RD0[o];
      Y1 += w * (KK_FLOAT)RD1[o];
    }

    e4  += L * Y;
    dL0 += L0 * Y;
    dL1 += L1 * Y;
    dL2 += L2 * Y;
    dL3 += L3 * Y;
    dR0 += L * Y0;
    dR1 += L * Y1;
  }

  if (eflag) energy += e4;

  KK_FLOAT fs[6];

  fs[canon_to_runtime[0]] = dL0 / dx[canon_to_runtime[0]];
  fs[canon_to_runtime[1]] = dL1 / dx[canon_to_runtime[1]];
  fs[canon_to_runtime[2]] = dL2 / dx[canon_to_runtime[2]];
  fs[canon_to_runtime[3]] = dL3 / dx[canon_to_runtime[3]];
  fs[canon_to_runtime[4]] = dR0 / dx[canon_to_runtime[4]];
  fs[canon_to_runtime[5]] = dR1 / dx[canon_to_runtime[5]];

  // ij
  force[0] += fs[0]*dr[0]; force[1] += fs[0]*dr[1]; force[2] += fs[0]*dr[2];
  force[CHDIM+0] -= fs[0]*dr[0]; force[CHDIM+1] -= fs[0]*dr[1]; force[CHDIM+2] -= fs[0]*dr[2];

  // ik
  force[0] += fs[1]*dr[CHDIM+0]; force[1] += fs[1]*dr[CHDIM+1]; force[2] += fs[1]*dr[CHDIM+2];
  force[2*CHDIM+0] -= fs[1]*dr[CHDIM+0]; force[2*CHDIM+1] -= fs[1]*dr[CHDIM+1]; force[2*CHDIM+2] -= fs[1]*dr[CHDIM+2];

  // il
  force[0] += fs[2]*dr[2*CHDIM+0]; force[1] += fs[2]*dr[2*CHDIM+1]; force[2] += fs[2]*dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fs[2]*dr[2*CHDIM+0]; force[3*CHDIM+1] -= fs[2]*dr[2*CHDIM+1]; force[3*CHDIM+2] -= fs[2]*dr[2*CHDIM+2];

  // jk
  force[CHDIM+0] += fs[3]*dr[3*CHDIM+0]; force[CHDIM+1] += fs[3]*dr[3*CHDIM+1]; force[CHDIM+2] += fs[3]*dr[3*CHDIM+2];
  force[2*CHDIM+0] -= fs[3]*dr[3*CHDIM+0]; force[2*CHDIM+1] -= fs[3]*dr[3*CHDIM+1]; force[2*CHDIM+2] -= fs[3]*dr[3*CHDIM+2];

  // jl
  force[CHDIM+0] += fs[4]*dr[4*CHDIM+0]; force[CHDIM+1] += fs[4]*dr[4*CHDIM+1]; force[CHDIM+2] += fs[4]*dr[4*CHDIM+2];
  force[3*CHDIM+0] -= fs[4]*dr[4*CHDIM+0]; force[3*CHDIM+1] -= fs[4]*dr[4*CHDIM+1]; force[3*CHDIM+2] -= fs[4]*dr[4*CHDIM+2];

  // kl
  force[2*CHDIM+0] += fs[5]*dr[5*CHDIM+0]; force[2*CHDIM+1] += fs[5]*dr[5*CHDIM+1]; force[2*CHDIM+2] += fs[5]*dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fs[5]*dr[5*CHDIM+0]; force[3*CHDIM+1] -= fs[5]*dr[5*CHDIM+1]; force[3*CHDIM+2] -= fs[5]*dr[5*CHDIM+2];

  if (vflag) {
    for (int p=0; p<6; p++) {
      stress[0] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+0];
      stress[1] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+1];
      stress[2] -= fs[p]*dr[p*CHDIM+0]*dr[p*CHDIM+2];
      stress[3] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+1];
      stress[4] -= fs[p]*dr[p*CHDIM+1]*dr[p*CHDIM+2];
      stress[5] -= fs[p]*dr[p*CHDIM+2]*dr[p*CHDIM+2];
    }
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[6];
  compute_4B(dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Structure containing temporary data.
  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.
  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  constexpr int natoms = 4;                     // Number of atoms in an interaction set
  constexpr int npairs = natoms*(natoms-1)/2;    // Number of pairs in an interaction set

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];
  KK_FLOAT deriv[npairs];

  KK_FLOAT Tn_ij[MAX_4B_POLY];
  KK_FLOAT Tn_ik[MAX_4B_POLY];
  KK_FLOAT Tn_il[MAX_4B_POLY];
  KK_FLOAT Tn_jk[MAX_4B_POLY];
  KK_FLOAT Tn_jl[MAX_4B_POLY];
  KK_FLOAT Tn_kl[MAX_4B_POLY];

  KK_FLOAT Tnd_ij[MAX_4B_POLY];
  KK_FLOAT Tnd_ik[MAX_4B_POLY];
  KK_FLOAT Tnd_il[MAX_4B_POLY];
  KK_FLOAT Tnd_jk[MAX_4B_POLY];
  KK_FLOAT Tnd_jl[MAX_4B_POLY];
  KK_FLOAT Tnd_kl[MAX_4B_POLY];

  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
      + typ_idxs[1]*natmtyps*natmtyps + typ_idxs[2]*natmtyps + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];

  //if (quadidx < 0) // Skipping an excluded interaction
  //  return;

  //auto c_mapped_pair_idx = c_pair_int_quad_map[idx];

  // Check whether cutoffs are within allowed ranges
/*
  for (int i=0; i<npairs; i++)
      if (dx[i] >= c_chimes_4b_cutoff(quadidx,1,c_pair_int_quad_map(idx,i)))
          return;
*/
  // These speed up fcut calculations by a LOT

  const KK_FLOAT cutoff_00 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,0),0);
  const KK_FLOAT cutoff_0 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,0),1);

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  const KK_FLOAT cutoff_01 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,1),0);
  const KK_FLOAT cutoff_1 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,1),1);

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  const KK_FLOAT cutoff_02 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,2),0);
  const KK_FLOAT cutoff_2 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,2),1);

  //if (dx[2] >= cutoff_2) // il
  //  return;

  const KK_FLOAT cutoff_03 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,3),0);
  const KK_FLOAT cutoff_3 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,3),1);

  //if (dx[3] >= cutoff_3) // jk
  //  return;

  const KK_FLOAT cutoff_04 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,4),0);
  const KK_FLOAT cutoff_4 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,4),1);

  //if (dx[4] >= cutoff_4) // jl
  //  return;

  const KK_FLOAT cutoff_05 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,5),0);
  const KK_FLOAT cutoff_5 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,5),1);

  //if (dx[5] >= cutoff_5) // kl
  //  return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  const int pair_type_1 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
  const int pair_type_2 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
  const int pair_type_3 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[3]);
  const int pair_type_4 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
  const int pair_type_5 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[3]);
  const int pair_type_6 = c_atom_int_pair_map(typ_idxs[2]*natmtyps + typ_idxs[3]);
  const int order = d_poly_orders[2];

  // Set up the polynomials

  set_cheby_polys(Tn_ij, Tnd_ij, dx[0], c_morse_var[pair_type_1], cutoff_00, cutoff_0, order);
  set_cheby_polys(Tn_ik, Tnd_ik, dx[1], c_morse_var[pair_type_2], cutoff_01, cutoff_1, order);
  set_cheby_polys(Tn_il, Tnd_il, dx[2], c_morse_var[pair_type_3], cutoff_02, cutoff_2, order);
  set_cheby_polys(Tn_jk, Tnd_jk, dx[3], c_morse_var[pair_type_4], cutoff_03, cutoff_3, order);
  set_cheby_polys(Tn_jl, Tnd_jl, dx[4], c_morse_var[pair_type_5], cutoff_04, cutoff_4, order);
  set_cheby_polys(Tn_kl, Tnd_kl, dx[5], c_morse_var[pair_type_6], cutoff_05, cutoff_5, order);

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  if (vflag)
    init_distance_tensor(dr2, dr, npairs);
#endif


  // Set up the smoothing functions
/*
  for (int i=0; i<npairs; i++)
      get_fcut(dx[i], c_chimes_4b_cutoff(quadidx,1,c_pair_int_quad_map(idx,i)], fcut[i], fcutderiv[i));
*/

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  get_fcut(dx[3], cutoff_3, fcut[3], fcutderiv[3]);
  get_fcut(dx[4], cutoff_4, fcut[4], fcutderiv[4]);
  get_fcut(dx[5], cutoff_5, fcut[5], fcutderiv[5]);

  // Product of all 6 fcuts

  const KK_FLOAT fcut_all = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];

  // Product of 5 fcuts

  KK_FLOAT fcut_5[npairs];
  fcut_5[0] = fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[1] = fcut[0] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[2] = fcut[0] * fcut[1] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[3] = fcut[0] * fcut[1] * fcut[2] * fcut[4] * fcut[5];
  fcut_5[4] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[5];
  fcut_5[5] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4];

  // Start the force/stress/energy calculation

  KK_FLOAT poly, dpoly_dx[npairs];

  //if (!dense_coeffs) {
  if (1) {
    poly_4B(poly, dpoly_dx, c_ncoeffs_4b[quadidx], quadidx, idx,
            Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik,
            Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else {
    // Dense evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[c_pair_int_quad_map(idx,j)] = j; }

    KK_FLOAT* Tn[npairs];
    KK_FLOAT* Tnd[npairs];

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij[0];
          Tnd[j] = &Tnd_ij[0];
          break;
        case 1:
          Tn[j] = &Tn_ik[0];
          Tnd[j] = &Tnd_ik[0];
          break;
        case 2:
          Tn[j] = &Tn_il[0];
          Tnd[j] = &Tnd_il[0];
          break;
        case 3:
          Tn[j] = &Tn_jk[0];
          Tnd[j] = &Tnd_jk[0];
          break;
        case 4:
          Tn[j] = &Tn_jl[0];
          Tnd[j] = &Tnd_jl[0];
          break;
        case 5:
          Tn[j] = &Tn_kl[0];
          Tnd[j] = &Tnd_kl[0];
          break;
        default:
          Kokkos::abort("Bad inverse pair mapping found");
      }
    }

    poly_4B_dense(poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], dpoly_dx[inv_mapped_pair[3]],
                  dpoly_dx[inv_mapped_pair[4]], dpoly_dx[inv_mapped_pair[5]], d_ncoeffs_4b[quadidx],
                  quadidx, Tn[0], Tn[1], Tn[2], Tn[3], Tn[4], Tn[5],
                  Tnd[0], Tnd[1], Tnd[2], Tnd[3], Tnd[4], Tnd[5]);
  }

  if (eflag)
    energy += poly * fcut_all;

  for (int j = 0; j < npairs; j++)
    force_scalar[j] = (fcut_all * dpoly_dx[j] + fcutderiv[j] * fcut_5[j] * poly) / dx[j];

  const KK_FLOAT &fscalar_0 = force_scalar[0];
  const KK_FLOAT &fscalar_1 = force_scalar[1];
  const KK_FLOAT &fscalar_2 = force_scalar[2];
  const KK_FLOAT &fscalar_3 = force_scalar[3];
  const KK_FLOAT &fscalar_4 = force_scalar[4];
  const KK_FLOAT &fscalar_5 = force_scalar[5];

  // Accumulate forces/stresses on/from the ij pair

  force[0] += fscalar_0 * dr[0];
  force[1] += fscalar_0 * dr[1];
  force[2] += fscalar_0 * dr[2];

  force[CHDIM+0] -= fscalar_0 * dr[0];
  force[CHDIM+1] -= fscalar_0 * dr[1];
  force[CHDIM+2] -= fscalar_0 * dr[2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_0 * dr2_4B(dr2,0,0,0,0); // xx tensor component
    stress[1] -= fscalar_0 * dr2_4B(dr2,0,0,0,1); // xy tensor component
    stress[2] -= fscalar_0 * dr2_4B(dr2,0,0,0,2); // xz tensor component
    stress[3] -= fscalar_0 * dr2_4B(dr2,0,1,0,1); // yy tensor component
    stress[4] -= fscalar_0 * dr2_4B(dr2,0,1,0,2); // yz tensor component
    stress[5] -= fscalar_0 * dr2_4B(dr2,0,2,0,2); // zz tensor component
#else
    stress[0] -= fscalar_0 * dr[0] * dr[0]; // xx tensor component
    stress[1] -= fscalar_0 * dr[0] * dr[1]; // xy tensor component
    stress[2] -= fscalar_0 * dr[0] * dr[2]; // xz tensor component
    stress[3] -= fscalar_0 * dr[1] * dr[1]; // yy tensor component
    stress[4] -= fscalar_0 * dr[1] * dr[2]; // yz tensor component
    stress[5] -= fscalar_0 * dr[2] * dr[2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0] += fscalar_1 * dr[CHDIM+0];
  force[1] += fscalar_1 * dr[CHDIM+1];
  force[2] += fscalar_1 * dr[CHDIM+2];
  force[2*CHDIM+0] -= fscalar_1 * dr[CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_1 * dr2_4B(dr2,1,0,1,0); // xx tensor component
    stress[1] -= fscalar_1 * dr2_4B(dr2,1,0,1,1); // xy tensor component
    stress[2] -= fscalar_1 * dr2_4B(dr2,1,0,1,2); // xz tensor component
    stress[3] -= fscalar_1 * dr2_4B(dr2,1,1,1,1); // yy tensor component
    stress[4] -= fscalar_1 * dr2_4B(dr2,1,1,1,2); // yz tensor component
    stress[5] -= fscalar_1 * dr2_4B(dr2,1,2,1,2); // zz tensor component
#else
    stress[0] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_1 * dr[CHDIM+2] * dr[CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the il pair

  force[0] += fscalar_2 * dr[2*CHDIM+0];
  force[1] += fscalar_2 * dr[2*CHDIM+1];
  force[2] += fscalar_2 * dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_2 * dr2_4B(dr2,2,0,2,0); // xx tensor component
    stress[1] -= fscalar_2 * dr2_4B(dr2,2,0,2,1); // xy tensor component
    stress[2] -= fscalar_2 * dr2_4B(dr2,2,0,2,2); // xz tensor component
    stress[3] -= fscalar_2 * dr2_4B(dr2,2,1,2,1); // yy tensor component
    stress[4] -= fscalar_2 * dr2_4B(dr2,2,1,2,2); // yz tensor component
    stress[5] -= fscalar_2 * dr2_4B(dr2,2,2,2,2); // zz tensor component
#else
    stress[0] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_2 * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jk pair

  force[CHDIM+0] += fscalar_3 * dr[3*CHDIM+0];
  force[CHDIM+1] += fscalar_3 * dr[3*CHDIM+1];
  force[CHDIM+2] += fscalar_3 * dr[3*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_3 * dr[3*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_3 * dr[3*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_3 * dr[3*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_3 * dr2_4B(dr2,3,0,3,0); // xx tensor component
    stress[1] -= fscalar_3 * dr2_4B(dr2,3,0,3,1); // xy tensor component
    stress[2] -= fscalar_3 * dr2_4B(dr2,3,0,3,2); // xz tensor component
    stress[3] -= fscalar_3 * dr2_4B(dr2,3,1,3,1); // yy tensor component
    stress[4] -= fscalar_3 * dr2_4B(dr2,3,1,3,2); // yz tensor component
    stress[5] -= fscalar_3 * dr2_4B(dr2,3,2,3,2); // zz tensor component
#else
    stress[0] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_3 * dr[3*CHDIM+1] * dr[3*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_3 * dr[3*CHDIM+1] * dr[3*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_3 * dr[3*CHDIM+2] * dr[3*CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jl pair

  force[CHDIM+0] += fscalar_4 * dr[4*CHDIM+0];
  force[CHDIM+1] += fscalar_4 * dr[4*CHDIM+1];
  force[CHDIM+2] += fscalar_4 * dr[4*CHDIM+2];

  force[3*CHDIM+0] -= fscalar_4 * dr[4*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_4 * dr[4*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_4 * dr[4*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_4 * dr2_4B(dr2,4,0,4,0); // xx tensor component
    stress[1] -= fscalar_4 * dr2_4B(dr2,4,0,4,1); // xy tensor component
    stress[2] -= fscalar_4 * dr2_4B(dr2,4,0,4,2); // xz tensor component
    stress[3] -= fscalar_4 * dr2_4B(dr2,4,1,4,1); // yy tensor component
    stress[4] -= fscalar_4 * dr2_4B(dr2,4,1,4,2); // yz tensor component
    stress[5] -= fscalar_4 * dr2_4B(dr2,4,2,4,2); // zz tensor component
#else
    stress[0] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_4 * dr[4*CHDIM+1] * dr[4*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_4 * dr[4*CHDIM+1] * dr[4*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_4 * dr[4*CHDIM+2] * dr[4*CHDIM+2]; // zz tensor component
#endif
  }
  // Accumulate forces/stresses on/from the kl pair

  force[2*CHDIM+0] += fscalar_5 * dr[5*CHDIM+0];
  force[2*CHDIM+1] += fscalar_5 * dr[5*CHDIM+1];
  force[2*CHDIM+2] += fscalar_5 * dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_5 * dr[5*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_5 * dr[5*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_5 * dr[5*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_5 * dr2_4B(dr2,5,0,5,0); // xx tensor component
    stress[1] -= fscalar_5 * dr2_4B(dr2,5,0,5,1); // xy tensor component
    stress[2] -= fscalar_5 * dr2_4B(dr2,5,0,5,2); // xz tensor component
    stress[3] -= fscalar_5 * dr2_4B(dr2,5,1,5,1); // yy tensor component
    stress[4] -= fscalar_5 * dr2_4B(dr2,5,1,5,2); // yz tensor component
    stress[5] -= fscalar_5 * dr2_4B(dr2,5,2,5,2); // zz tensor component
#else
    stress[0] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_5 * dr[5*CHDIM+1] * dr[5*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_5 * dr[5*CHDIM+1] * dr[5*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_5 * dr[5*CHDIM+2] * dr[5*CHDIM+2]; // zz tensor component
#endif
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void chimesFFKokkos<DeviceType>::build_pair_int_quad_map()
// Build the pair maps for all possible quads.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  chimesFF::build_pair_int_quad_map();

  // pair_int_quad_map

  int size = pair_int_quad_map.size();
  int max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,pair_int_quad_map[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_pair_int_quad_map,"chimesFF:pair_int_quad_map",size,max_j);

  auto h_pair_int_quad_map = Kokkos::create_mirror_view(d_pair_int_quad_map);

  for (int i = 0; i < size; i++) {
    int size_j = pair_int_quad_map[i].size();
    for (int j = 0; j < size_j; j++) {
      h_pair_int_quad_map(i,j) = pair_int_quad_map[i][j];
    }
  }

  Kokkos::deep_copy(d_pair_int_quad_map,h_pair_int_quad_map);
  c_pair_int_quad_map = d_pair_int_quad_map;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void chimesFFKokkos<DeviceType>::build_pair_int_trip_map()
// Build the pair maps for all possible triplets.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  chimesFF::build_pair_int_trip_map();

  // pair_int_trip_map

  int size = pair_int_trip_map.size();
  int max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,pair_int_trip_map[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_pair_int_trip_map,"chimesFF:pair_int_trip_map",size,max_j);

  auto h_pair_int_trip_map = Kokkos::create_mirror_view(d_pair_int_trip_map);

  for (int i = 0; i < size; i++) {
    int size_j = pair_int_trip_map[i].size();
    for (int j = 0; j < size_j; j++) {
      h_pair_int_trip_map(i,j) = pair_int_trip_map[i][j];
    }
  }

  Kokkos::deep_copy(d_pair_int_trip_map,h_pair_int_trip_map);
  c_pair_int_trip_map = d_pair_int_trip_map;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_2B(KK_FLOAT &e, KK_FLOAT &f0, const int ncoeffs_2b, const int pair_idx,
                                         KK_FLOAT* Tn, KK_FLOAT* Tnd) const
// Compute the 2 body polynomial (e) and derivatives with respect to the pair distance (f0)
// (LEF) 3/11/26
{
  e = 0.0;
  f0 = 0.0;

  auto c_chimes_2b_params_pairidx = Kokkos::subview(c_chimes_2b_params,pair_idx,Kokkos::ALL);
  auto c_chimes_2b_pows_pairidx = Kokkos::subview(c_chimes_2b_pows,pair_idx,Kokkos::ALL);

  #pragma unroll
  for (int coeffs = 0; coeffs < ncoeffs_2b; coeffs++) {
    const KK_FLOAT coeff_val = c_chimes_2b_params_pairidx(coeffs);
    const int powerp1 = c_chimes_2b_pows_pairidx(coeffs) + 1;

    e += coeff_val * Tn[powerp1];

    f0 += coeff_val * Tnd[powerp1];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B(KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
                                         KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk,
                                         KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{

  e = 0.0;
  f[0] = 0.0;
  f[1] = 0.0;
  f[2] = 0.0;

  int trip_map_idx[3];
  trip_map_idx[0] = c_pair_int_trip_map(idx,0);
  trip_map_idx[1] = c_pair_int_trip_map(idx,1);
  trip_map_idx[2] = c_pair_int_trip_map(idx,2);

  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);
  auto c_chimes_3b_powers_tripidx = Kokkos::subview(c_chimes_3b_powers,tripidx,Kokkos::ALL,Kokkos::ALL);

  #pragma unroll
  for (int coeffs = 0; coeffs < ncoeffs_3b; coeffs++) {
    const KK_FLOAT coeff = c_chimes_3b_params_tripidx(coeffs);

    const int powers[3] = { c_chimes_3b_powers_tripidx(coeffs,trip_map_idx[0]), 
                      c_chimes_3b_powers_tripidx(coeffs,trip_map_idx[1]), 
                      c_chimes_3b_powers_tripidx(coeffs,trip_map_idx[2]) };

    e += coeff * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk[powers[2]];

    f[0] += coeff * Tnd_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk[powers[2]];
    f[1] += coeff * Tnd_ik[powers[1]] * Tn_ij[powers[0]] * Tn_jk[powers[2]];
    f[2] += coeff * Tnd_jk[powers[2]] * Tn_ij[powers[0]] * Tn_ik[powers[1]];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense(KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, int ncoeffs_3b, int tripidx,
                                               KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik, KK_FLOAT *Tn_jk,
                                               KK_FLOAT *Tnd_ij, KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
  const int loop_style = 2;

  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;

  if (ncoeffs_3b == 0) return;

  int max_poly = 0;
  const int loop_max = 1000;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i == ncoeffs_3b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    Kokkos::abort("Bad number of 3 body coefficients for dense evaluation");
  }

  if (loop_style == 1) {
    poly_3B_dense_loop1(max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  } else if (loop_style == 2) {
    poly_3B_dense_loop2(max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  //} else if (loop_style == 3) {
  //  poly_3B_dense_loop3(max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
  //                      Tnd_ij, Tnd_ik, Tnd_jk);
  } else {
    Kokkos::abort("Error: bad 3 body dense loop style");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                                   int ncoeffs_3b, int tripidx,
                                   KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik,
                                   KK_FLOAT *Tn_jk, KK_FLOAT *Tnd_ij,
                                   KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
{
  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);

  #pragma unroll
  for (int count = 0; count < ncoeffs_3b; count++) {
    int l = count / (max_poly * max_poly);
    //if (l >= max_poly) { cout << "Internal error: l > max_poly: " << l << "\n"; }
    int m = (count / max_poly) % max_poly;
    int n = count % max_poly;

    const KK_FLOAT coeff = c_chimes_3b_params_tripidx[count];
    if (coeff != 0.0) {
      const KK_FLOAT tn_ij = Tn_ij[l];
      const KK_FLOAT tnd_ij = Tnd_ij[l];
      const KK_FLOAT tn_ik = Tn_ik[m];
      const KK_FLOAT tnd_ik = Tnd_ik[m];
      const KK_FLOAT tn_jk = Tn_jk[n];
      const KK_FLOAT tnd_jk = Tnd_jk[n];

      e += coeff * tn_ij * tn_ik * tn_jk;
      f0 += coeff * tnd_ij * tn_ik * tn_jk;
      f1 += coeff * tnd_ik * tn_ij * tn_jk;
      f2 += coeff * tnd_jk * tn_ij * tn_ik;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense_loop2(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                                   int ncoeffs_3b, int tripidx,
                                   KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik,
                                   KK_FLOAT *Tn_jk, KK_FLOAT *Tnd_ij,
                                   KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
{
  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);

  int count = 0;
  for (int i = 0; i < max_poly; i++) {
    const KK_FLOAT tn_ij = Tn_ij[i];
    const KK_FLOAT tnd_ij = Tnd_ij[i];

    for (int j = 0; j < max_poly; j++) {
      const KK_FLOAT tn_ik = Tn_ik[j];
      const KK_FLOAT tnd_ik = Tnd_ik[j];
      const KK_FLOAT tn_ij_ik = tn_ij * tn_ik;

      for (int k = 0; k < max_poly; k++) {
        const KK_FLOAT coeff = c_chimes_3b_params_tripidx[count];
        if (coeff != 0.0) {
          const KK_FLOAT tn_jk = Tn_jk[k];
          const KK_FLOAT tnd_jk = Tnd_jk[k];

          e += coeff * tn_ij_ik * tn_jk;
          f0 += coeff * tnd_ij * tn_ik * tn_jk;
          f1 += coeff * tnd_ik * tn_ij * tn_jk;
          f2 += coeff * tnd_jk * tn_ij_ik;
        }
        count++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B(KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
                                         KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                                         KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                                         KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                                         KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  constexpr int npairs = 6;
  int powers[npairs],quad_map_idx[npairs];

  e = 0;
  for (int i = 0; i < npairs; i++) f[i] = 0.0;

  for (int i = 0; i < npairs; i++) quad_map_idx[i] = c_pair_int_quad_map(idx,i);

  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);
  auto c_chimes_4b_powers_quadidx = Kokkos::subview(c_chimes_4b_powers,quadidx,Kokkos::ALL,Kokkos::ALL);

  #pragma unroll
  for (int coeffs = 0; coeffs < ncoeffs_4b; coeffs++) {
    const KK_FLOAT coeff = c_chimes_4b_params_quadidx(coeffs);

    for (int i = 0; i < npairs; i++) powers[i] = c_chimes_4b_powers_quadidx(coeffs,quad_map_idx[i]);

    const KK_FLOAT Tn_ij_ik_il = Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_il[powers[2]];
    const KK_FLOAT Tn_jk_jl = Tn_jk[powers[3]] * Tn_jl[powers[4]];
    const KK_FLOAT Tn_kl_5 = Tn_kl[powers[5]];

    e += coeff * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5;

    f[0] += coeff * Tnd_ij[powers[0]] * Tn_ik[powers[1]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
    f[1] += coeff * Tnd_ik[powers[1]] * Tn_ij[powers[0]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
    f[2] += coeff * Tnd_il[powers[2]] * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk_jl * Tn_kl_5;
    f[3] += coeff * Tnd_jk[powers[3]] * Tn_ij_ik_il * Tn_jl[powers[4]] * Tn_kl_5;
    f[4] += coeff * Tnd_jl[powers[4]] * Tn_ij_ik_il * Tn_jk[powers[3]] * Tn_kl_5;
    f[5] += coeff * Tnd_kl[powers[5]] * Tn_ij_ik_il * Tn_jk_jl;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense(KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4,
                             KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                             KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                             KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                             KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                             KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
  const int loop_style = 2;

  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;
  f3 = 0.0;
  f4 = 0.0;
  f5 = 0.0;

  if (ncoeffs_4b == 0) return;

  int max_poly = 0;
  const int loop_max = 100;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i * i * i * i == ncoeffs_4b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    Kokkos::abort("Bad number of 4 body coefficients for dense evaluation");
  }

  if (loop_style == 1) {
    poly_4B_dense_loop1(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, quadidx, Tn_ij, Tn_ik,
                        Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else if (loop_style == 2) {
    poly_4B_dense_loop2(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, quadidx, Tn_ij, Tn_ik,
                        Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  //} else if (loop_style == 3) {
  //  poly_4B_dense_loop3(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, quadidx, Tn_ij, Tn_ik,
  //                      Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense_loop1(
    int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4, KK_FLOAT &f5,
    int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
    KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
    KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
    KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
{
  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);

  int max_poly_pow[6];
  max_poly_pow[5] = 1;

  for (int l = 4; l >= 0; l--) { max_poly_pow[l] = max_poly_pow[l + 1] * max_poly; }

  #pragma unroll
  for (int count = 0; count < ncoeffs_4b; count++) {
    if (c_chimes_4b_params_quadidx[count] != 0.0) {
      int index[6];
      for (int i = 0; i < 6; i++) { index[i] = (count / max_poly_pow[i]) % max_poly; }
      const KK_FLOAT tn_ij = Tn_ij[index[0]];
      const KK_FLOAT tnd_ij = Tnd_ij[index[0]];
      const KK_FLOAT tn_ik = Tn_ik[index[1]];
      const KK_FLOAT tnd_ik = Tnd_ik[index[1]];
      const KK_FLOAT tn_il = Tn_il[index[2]];
      const KK_FLOAT tnd_il = Tnd_il[index[2]];
      const KK_FLOAT tn_jk = Tn_jk[index[3]];
      const KK_FLOAT tnd_jk = Tnd_jk[index[3]];
      const KK_FLOAT tn_jl = Tn_jl[index[4]];
      const KK_FLOAT tnd_jl = Tnd_jl[index[4]];
      const KK_FLOAT tn_kl = Tn_kl[index[5]];
      const KK_FLOAT tnd_kl = Tnd_kl[index[5]];
      const KK_FLOAT coeff = c_chimes_4b_params_quadidx[count];

      const KK_FLOAT Tn_jk_jl = tn_jk * tn_jl;
      const KK_FLOAT Tn_ij_ik_il = tn_ij * tn_ik * tn_il;

      e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

      f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;
      f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;
      f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;
      f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;
      f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;
      f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense_loop2(
    int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4, KK_FLOAT &f5,
    int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
    KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
    KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
    KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
{
  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);

  int count = 0;
  for (int i = 0; i < max_poly; i++) {
    const KK_FLOAT tn_ij = Tn_ij[i];
    const KK_FLOAT tnd_ij = Tnd_ij[i];
    for (int j = 0; j < max_poly; j++) {
      const KK_FLOAT tn_ik = Tn_ik[j];
      const KK_FLOAT tnd_ik = Tnd_ik[j];
      for (int l = 0; l < max_poly; l++) {
        const KK_FLOAT tn_il = Tn_il[l];
        const KK_FLOAT tnd_il = Tnd_il[l];
        const KK_FLOAT Tn_ij_ik_il = tn_ij * tn_ik * tn_il;
        for (int m = 0; m < max_poly; m++) {
          const KK_FLOAT tn_jk = Tn_jk[m];
          const KK_FLOAT tnd_jk = Tnd_jk[m];
          for (int n = 0; n < max_poly; n++) {
            const KK_FLOAT tn_jl = Tn_jl[n];
            const KK_FLOAT tnd_jl = Tnd_jl[n];
            const KK_FLOAT Tn_jk_jl = tn_jk * tn_jl;
            for (int o = 0; o < max_poly; o++) {
              const KK_FLOAT tn_kl = Tn_kl[o];
              const KK_FLOAT tnd_kl = Tnd_kl[o];

              if (c_chimes_4b_params_quadidx[count] != 0.0) {
                const KK_FLOAT coeff = c_chimes_4b_params_quadidx[count];

                e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

                f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;
                f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;
                f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;
                f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;
                f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;
                f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
              }
              count++;
            }
          }
        }
      }
    }
  }
}