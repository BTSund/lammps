/*
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author: Stan Moore (2025)
*/

#ifndef _chimesFF_KOKKOS_h
#define _chimesFF_KOKKOS_h

#include "chimesFF.h"
#include "kokkos_type.h"
#include "memory_kokkos.h"

#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

#define pi 3.14159265359

constexpr int MAX_2B_POLY = 40;
constexpr int MAX_3B_POLY = 20;
constexpr int MAX_4B_POLY = 10;

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys)
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)


#define CHDIM 3 // The number of spatial dimensions.
#ifdef USE_DISTANCE_TENSOR
#undef USE_DISTANCE_TENSOR
#endif
#define USE_DISTANCE_TENSOR 0 // Use tensor of distances in computing stresses.

template<class DeviceType>
class chimesFFKokkos : public chimesFF
{
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  ////////////////////////
  // General parameters
  ////////////////////////

  int d_poly_orders[3];    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order


  bool tabulate_4B_svd3x3_corner_kk = false;
  bool tabulate_4B_svd4x2_corner_kk = false;
  bool tabulate_2B_kk = false;
  bool tabulate_3B_kk = false;
  bool tabulate_4B_svd3x3_kk = false;
  bool tabulate_4B_svd4x2_kk = false;

  bool tabulate_4B_cp_kk = false;
  bool tabulate_4B_cp_direct_kk = false;
  bool tabulate_3B_cp_kk = false;

  // CP 1D tables on the device (precision set by chimes_tab_real in chimesFF.h)
  typedef Kokkos::View<chimes_tab_real*, DeviceType> t_cp_tab_1d;

  // ------------------------------------------------------------------
  // CP 1D 4B tabulation, Kokkos packed
  // ------------------------------------------------------------------

  typename AT::t_int_1d d_tab4b_cp_rank;
  typename AT::t_int_1d_const_um c_tab4b_cp_rank;

  typename AT::t_int_2d d_tab4b_cp_canon_to_param; // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_canon_to_param;

  typename AT::t_int_2d d_tab4b_cp_ngrid;          // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_ngrid;

  typename AT::t_int_2d d_tab4b_cp_offset;         // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_offset;

  typename AT::t_kkfloat_2d d_tab4b_cp_r0;         // [quad][6]
  typename AT::t_kkfloat_2d_const_um c_tab4b_cp_r0;

  typename AT::t_kkfloat_2d d_tab4b_cp_dr;         // [quad][6]
  typename AT::t_kkfloat_2d_const_um c_tab4b_cp_dr;

  typename AT::t_kkfloat_2d d_tab4b_cp_invdr;      // [quad][6]
  typename AT::t_kkfloat_2d_const_um c_tab4b_cp_invdr;

  // Interleaved like the host tables: for a slot starting at offset off,
  //   vd[off + row*2*R + q]     = value      of rank term q at row
  //   vd[off + row*2*R + R + q] = derivative of rank term q at row
  t_cp_tab_1d d_tab4b_cp_vd;

  // Cubic-Hermite weights for one CP 1D table lookup (3B and 4B), with the
  // exact continuation below the first grid point; returns the row index.
  KOKKOS_INLINE_FUNCTION
  int cp1d_weights(const KK_FLOAT r, const KK_FLOAT r0, const KK_FLOAT h, const KK_FLOAT invh, const int ngrid, const KK_FLOAT rc, KK_FLOAT* wv, KK_FLOAT* wd) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B_cp_tab(
      const KK_FLOAT* dx,
      const KK_FLOAT* dr,
      const int* typ_idxs,
      KK_FLOAT* force,
      KK_FLOAT* stress,
      KK_FLOAT& energy) const;

  // ------------------------------------------------------------------
  // Direct/on-the-fly CP 4B factors, Kokkos packed
  // ------------------------------------------------------------------

  typename AT::t_int_1d d_tab4b_cp_direct_rank;
  typename AT::t_int_1d_const_um c_tab4b_cp_direct_rank;

  typename AT::t_int_2d d_tab4b_cp_direct_canon_to_param; // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_direct_canon_to_param;

  typename AT::t_int_2d d_tab4b_cp_direct_orders;         // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_direct_orders;

  typename AT::t_int_2d d_tab4b_cp_direct_offset;         // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_cp_direct_offset;

  typename AT::t_kkfloat_1d d_tab4b_cp_direct_factor;
  typename AT::t_kkfloat_1d_const_um c_tab4b_cp_direct_factor;

  KOKKOS_INLINE_FUNCTION
  void compute_4B_cp_direct(
      const KK_FLOAT* dx,
      const KK_FLOAT* dr,
      const int* typ_idxs,
      KK_FLOAT* force,
      KK_FLOAT* stress,
      KK_FLOAT& energy) const;

  // ------------------------------------------------------------------
  // CP 1D 3B tabulation, Kokkos packed (same layout as the 4B CP tables;
  // a shared symmetric table is stored once and used by all three slots)
  // ------------------------------------------------------------------

  typename AT::t_int_1d d_tab3b_cp_rank;            // [trip]
  typename AT::t_int_1d_const_um c_tab3b_cp_rank;

  typename AT::t_int_2d d_tab3b_cp_canon_to_param;  // [trip][3]
  typename AT::t_int_2d_const_um c_tab3b_cp_canon_to_param;

  typename AT::t_int_2d d_tab3b_cp_ngrid;           // [trip][3]
  typename AT::t_int_2d_const_um c_tab3b_cp_ngrid;

  typename AT::t_int_2d d_tab3b_cp_offset;          // [trip][3]
  typename AT::t_int_2d_const_um c_tab3b_cp_offset;

  typename AT::t_kkfloat_2d d_tab3b_cp_r0;          // [trip][3]
  typename AT::t_kkfloat_2d_const_um c_tab3b_cp_r0;

  typename AT::t_kkfloat_2d d_tab3b_cp_dr;          // [trip][3]
  typename AT::t_kkfloat_2d_const_um c_tab3b_cp_dr;

  typename AT::t_kkfloat_2d d_tab3b_cp_invdr;       // [trip][3]
  typename AT::t_kkfloat_2d_const_um c_tab3b_cp_invdr;

  t_cp_tab_1d d_tab3b_cp_vd;

  KOKKOS_INLINE_FUNCTION
  void compute_3B_cp_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const;

    typename AT::t_int_1d d_tab2b_n;
  typename AT::t_int_1d_const_um c_tab2b_n;

  typename AT::t_kkfloat_1d d_tab2b_r0;
  typename AT::t_kkfloat_1d_const_um c_tab2b_r0;

  typename AT::t_kkfloat_1d d_tab2b_dr;
  typename AT::t_kkfloat_1d_const_um c_tab2b_dr;

  typename AT::t_kkfloat_1d d_tab2b_invdr;
  typename AT::t_kkfloat_1d_const_um c_tab2b_invdr;

  typename AT::t_int_1d d_tab2b_offset;
  typename AT::t_int_1d_const_um c_tab2b_offset;

  typename Kokkos::View<float*> d_tab2b_e;
  typename Kokkos::View<float*, typename AT::t_kkfloat_1d::array_layout, device_type,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > c_tab2b_e;

  typename Kokkos::View<float*> d_tab2b_f;
  typename Kokkos::View<float*, typename AT::t_kkfloat_1d::array_layout, device_type,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > c_tab2b_f;




    typename AT::t_int_1d d_tab3b_n;
  typename AT::t_int_1d_const_um c_tab3b_n;

  typename AT::t_kkfloat_1d d_tab3b_r0;
  typename AT::t_kkfloat_1d_const_um c_tab3b_r0;

  typename AT::t_kkfloat_1d d_tab3b_dr;
  typename AT::t_kkfloat_1d_const_um c_tab3b_dr;

  typename AT::t_kkfloat_1d d_tab3b_invdr;
  typename AT::t_kkfloat_1d_const_um c_tab3b_invdr;

  typename AT::t_int_1d d_tab3b_offset;
  typename AT::t_int_1d_const_um c_tab3b_offset;

  // map canonical slot -> parameter slot for each tripidx, length 3
  typename AT::t_int_2d d_tab3b_canon_to_param;
  typename AT::t_int_2d_const_um c_tab3b_canon_to_param;

  Kokkos::View<float*> d_tab3b_e;
  Kokkos::View<float*> d_tab3b_f0;
  Kokkos::View<float*> d_tab3b_f1;
  Kokkos::View<float*> d_tab3b_f2;

  typename AT::t_int_1d d_tab4b_svd3_rank;
  typename AT::t_int_1d_const_um c_tab4b_svd3_rank;

  typename AT::t_int_1d d_tab4b_svd3_ngrid;
  typename AT::t_int_1d_const_um c_tab4b_svd3_ngrid;

  typename AT::t_int_1d d_tab4b_svd3_offset_left;
  typename AT::t_int_1d_const_um c_tab4b_svd3_offset_left;

  typename AT::t_int_1d d_tab4b_svd3_offset_right;
  typename AT::t_int_1d_const_um c_tab4b_svd3_offset_right;

  typename AT::t_kkfloat_2d d_tab4b_svd3_r0_left;    // [quad][3]
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_r0_left;

  typename AT::t_kkfloat_2d d_tab4b_svd3_dr_left;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_dr_left;

  typename AT::t_kkfloat_2d d_tab4b_svd3_invdr_left;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_invdr_left;

  typename AT::t_kkfloat_2d d_tab4b_svd3_r0_right;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_r0_right;

  typename AT::t_kkfloat_2d d_tab4b_svd3_dr_right;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_dr_right;

  typename AT::t_kkfloat_2d d_tab4b_svd3_invdr_right;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd3_invdr_right;

  typename AT::t_int_2d d_tab4b_svd3_canon_to_param; // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_svd3_canon_to_param;

  Kokkos::View<float*> d_tab4b_svd3_left_val;
  Kokkos::View<float*> d_tab4b_svd3_left_d0;
  Kokkos::View<float*> d_tab4b_svd3_left_d1;
  Kokkos::View<float*> d_tab4b_svd3_left_d2;

  Kokkos::View<float*> d_tab4b_svd3_right_val;
  Kokkos::View<float*> d_tab4b_svd3_right_d0;
  Kokkos::View<float*> d_tab4b_svd3_right_d1;
  Kokkos::View<float*> d_tab4b_svd3_right_d2;


    typename AT::t_int_1d d_tab4b_svd42_rank;
  typename AT::t_int_1d_const_um c_tab4b_svd42_rank;

  typename AT::t_int_1d d_tab4b_svd42_ngrid_left;
  typename AT::t_int_1d_const_um c_tab4b_svd42_ngrid_left;

  typename AT::t_int_1d d_tab4b_svd42_ngrid_right;
  typename AT::t_int_1d_const_um c_tab4b_svd42_ngrid_right;

  typename AT::t_int_1d d_tab4b_svd42_offset_left;
  typename AT::t_int_1d_const_um c_tab4b_svd42_offset_left;

  typename AT::t_int_1d d_tab4b_svd42_offset_right;
  typename AT::t_int_1d_const_um c_tab4b_svd42_offset_right;

  typename AT::t_kkfloat_2d d_tab4b_svd42_r0_left;    // [quad][4]
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_r0_left;

  typename AT::t_kkfloat_2d d_tab4b_svd42_dr_left;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_dr_left;

  typename AT::t_kkfloat_2d d_tab4b_svd42_invdr_left;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_invdr_left;

  typename AT::t_kkfloat_2d d_tab4b_svd42_r0_right;   // [quad][2]
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_r0_right;

  typename AT::t_kkfloat_2d d_tab4b_svd42_dr_right;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_dr_right;

  typename AT::t_kkfloat_2d d_tab4b_svd42_invdr_right;
  typename AT::t_kkfloat_2d_const_um c_tab4b_svd42_invdr_right;

  typename AT::t_int_2d d_tab4b_svd42_canon_to_param; // [quad][6]
  typename AT::t_int_2d_const_um c_tab4b_svd42_canon_to_param;

  Kokkos::View<float*> d_tab4b_svd42_left_val;
  Kokkos::View<float*> d_tab4b_svd42_left_d0;
  Kokkos::View<float*> d_tab4b_svd42_left_d1;
  Kokkos::View<float*> d_tab4b_svd42_left_d2;
  Kokkos::View<float*> d_tab4b_svd42_left_d3;

  Kokkos::View<float*> d_tab4b_svd42_right_val;
  Kokkos::View<float*> d_tab4b_svd42_right_d0;
  Kokkos::View<float*> d_tab4b_svd42_right_d1;

  static constexpr int MAX_SVD_RANK = 256;


  KOKKOS_INLINE_FUNCTION
  KK_FLOAT interp_tab2b_linear(int pair_idx, KK_FLOAT r, const float *tab) const;

  KOKKOS_INLINE_FUNCTION
  void compute_2B_tab(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs,
                      KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const;


  KOKKOS_INLINE_FUNCTION
  void interp_tab3b_linear(int tripidx, const KK_FLOAT *rquery,
                           KK_FLOAT &e, KK_FLOAT &d0, KK_FLOAT &d1, KK_FLOAT &d2) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs,
                      KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const;


    KOKKOS_INLINE_FUNCTION
  void interp_svd3x3_side_linear(
      int quadidx, bool left_side, const KK_FLOAT *rquery,
      KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1, KK_FLOAT *d2) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B_svd3x3_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs,
                             KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const;


    KOKKOS_INLINE_FUNCTION
  void interp_svd4x2_left_linear(
      int quadidx, const KK_FLOAT *rquery,
      KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1, KK_FLOAT *d2, KK_FLOAT *d3) const;

  KOKKOS_INLINE_FUNCTION
  void interp_svd4x2_right_linear(
      int quadidx, const KK_FLOAT *rquery,
      KK_FLOAT *val, KK_FLOAT *d0, KK_FLOAT *d1) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B_svd4x2_tab(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs,
                             KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT& energy) const;
  ////////////////////////
  // Functions
  ////////////////////////

  chimesFFKokkos();
  ~chimesFFKokkos();

  void read_parameters(string paramfile) override;

  // Frees the host copies of tables already packed into device views
  void release_host_tables();

  KOKKOS_INLINE_FUNCTION
  void compute_1B(const int typ_idx, KK_FLOAT & energy) const;

  // 2+B compute functions overloaded with force_scalar var for compatibility with LAMMPS

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force,KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const;

  // Functions to aid using ChIMES Calculator for fitting

  void build_pair_int_trip_map() override;
  void build_pair_int_quad_map() override;

  int eflag,vflag;

 private:

  typename AT::t_kkfloat_1d d_morse_var;      // [npairs]; morse_lambda
  typename AT::t_kkfloat_1d_const_um c_morse_var;

  KK_FLOAT d_penalty_params[2]; // [2];  Second dimension: [0] = A_pen, [1] = d_pen
  typename AT::t_kkfloat_1d d_energy_offsets; // [natmtyps]; Single atom ChIMES energies
  typename AT::t_kkfloat_1d_const_um c_energy_offsets;

 public:

  ////////////////////////
  // Definitions for pair, triplet, and quadruplet types
  ////////////////////////

  // 2-body maps

  typename AT::t_int_1d d_atom_int_pair_map; // [nmaps] "fast" maps, based on atom type index
  typename AT::t_int_1d_const_um c_atom_int_pair_map;

  // 3-body maps

  typename AT::t_int_1d d_atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
  typename AT::t_int_1d_const_um c_atom_int_trip_map;

  typename AT::t_int_2d d_pair_int_trip_map;  // Gives the atom pair indices for an arbitrary triplet of atom types.
  typename AT::t_int_2d_const_um c_pair_int_trip_map;

  // 4-body maps

  typename AT::t_int_1d d_atom_int_quad_map; // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
  typename AT::t_int_1d_const_um c_atom_int_quad_map;

  typename AT::t_int_2d d_pair_int_quad_map;  // Gives the atom pair indices for an arbitrary quad of atom types.
  typename AT::t_int_2d_const_um c_pair_int_quad_map;

  ////////////////////////
  // Polynomial parameters
  ////////////////////////

  // number of coefficients for the pair/triplet/quadruplet type

  typename AT::t_int_1d d_ncoeffs_2b;        // [npairs]
  typename AT::t_int_1d_const_um c_ncoeffs_2b;

  typename AT::t_int_2d d_chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter
  typename AT::t_int_2d_const_um c_chimes_2b_pows;

  typename AT::t_kkfloat_2d d_chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients
  typename AT::t_kkfloat_2d_const_um c_chimes_2b_params;

  typename AT::t_kkfloat_1d_2 d_chimes_2b_cutoff;  // [npairs][2] inner and outer cutoff for pair
  typename AT::t_kkfloat_1d_2_const_um c_chimes_2b_cutoff;

  typename AT::t_int_1d d_ncoeffs_3b;              // [ntrips]
  typename AT::t_int_1d_const_um c_ncoeffs_3b;

  typename AT::t_int_3d d_chimes_3b_powers;    // [ntrips][nparams][constit. pair]
  typename AT::t_int_3d_const_um c_chimes_3b_powers;

  typename AT::t_kkfloat_2d d_chimes_3b_params;    // [ntrips][nparams]
  typename AT::t_kkfloat_2d_const_um c_chimes_3b_params;

  typename AT::t_kkfloat_3d d_chimes_3b_cutoff;    // [ntrips][constit. pair][2] inner and outer cutoff for pair 1
  typename AT::t_kkfloat_3d_const_um c_chimes_3b_cutoff;

  typename AT::t_int_1d d_ncoeffs_4b;          // [nquads]
  typename AT::t_int_1d_const_um c_ncoeffs_4b;

  typename AT::t_int_3d d_chimes_4b_powers;    // [nquads][nparams][constit. pair]
  typename AT::t_int_3d_const_um c_chimes_4b_powers;

  typename AT::t_kkfloat_2d d_chimes_4b_params;    // [nquads][nparams]
  typename AT::t_kkfloat_2d_const_um c_chimes_4b_params;

  typename AT::t_kkfloat_3d d_chimes_4b_cutoff;    // [nquads][constit. pair][2] inner and outer cutoff for pair 1
  typename AT::t_kkfloat_3d_const_um c_chimes_4b_cutoff;

 private:

  // Tools for compute functions

  KOKKOS_INLINE_FUNCTION
  void set_cheby_polys(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                       const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const;

  KOKKOS_INLINE_FUNCTION
  void poly_2B(KK_FLOAT &e, KK_FLOAT &f0, int ncoeffs_2b, int pair_idx,
               KK_FLOAT* Tn, KK_FLOAT* Tnd) const;

  KOKKOS_INLINE_FUNCTION
  void poly_3B(KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
               KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk,
               KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  // Evaluates the 3-Body chebyshev polynomial in dense format

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense(KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, int ncoeffs_3b,
                     int tripidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
                     KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik,
                     KK_FLOAT* Tnd_jk) const;

  // Loop evaluators for poly_3B_dense

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                           int ncoeffs_3b, int tripidx, KK_FLOAT* Tn_ij,
                           KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij,
                           KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense_loop2(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                           int ncoeffs_3b, int tripidx, KK_FLOAT* Tn_ij,
                           KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij,
                           KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  KOKKOS_INLINE_FUNCTION
  void poly_4B(KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
               KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
               KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
               KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
               KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Evaluates the 4-body Chebyshev polynomial in dense format

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense(KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4,
                     KK_FLOAT &f5, int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij,
                     KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk,
                     KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl, KK_FLOAT* Tnd_ij,
                     KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
                     KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Loop1 uses a flat loop to evaluate a dense 4-body polynomial

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3,
                           KK_FLOAT &f4, KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                           KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                           KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                           KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                           KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Innver evaluation loop for dense 4 body poly.  2nd. variant.

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense_loop2(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3,
                           KK_FLOAT &f4, KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                           KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                           KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                           KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                           KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  KOKKOS_INLINE_FUNCTION
  void set_polys_out_of_range(KK_FLOAT* d_Tn, KK_FLOAT* d_Tnd, KK_FLOAT dx, KK_FLOAT x,
                              int poly_order, KK_FLOAT inner_cutoff, KK_FLOAT exprlen, KK_FLOAT dx_dr) const;

  KOKKOS_INLINE_FUNCTION
  void get_fcut(const KK_FLOAT dx, const KK_FLOAT outer_cutoff, KK_FLOAT & fcut, KK_FLOAT & fcutderiv) const;

  KOKKOS_INLINE_FUNCTION
  void get_penalty(const KK_FLOAT dx, const int & pair_idx, KK_FLOAT & E_penalty, KK_FLOAT & force_scalar) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT dr2_3B(const KK_FLOAT* dr2, int i, int j, int k, int l) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT dr2_4B(const KK_FLOAT* dr2, int i, int j, int k, int l) const;

  KOKKOS_INLINE_FUNCTION
  void init_distance_tensor(KK_FLOAT* dr2, const KK_FLOAT* dr, int natoms) const;
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::get_fcut(const KK_FLOAT dx, const KK_FLOAT outer_cutoff, KK_FLOAT & fcut, KK_FLOAT & fcutderiv) const
{
  KK_FLOAT fcut0;
  KK_FLOAT fcut0_deriv;

  if (fcut_type == fcutType::CUBIC) {
    fcut0 = (1.0 - dx/outer_cutoff);
    fcut = pow(fcut0,3.0);
    fcutderiv = pow(fcut0,2.0);
    fcutderiv *= -1.0 * 3.0 /outer_cutoff;
  } else if (fcut_type == fcutType::TERSOFF) {

    KK_FLOAT THRESH = outer_cutoff-fcut_var*outer_cutoff;

    if (dx < THRESH)        // Case 1: Our pair distance is less than the fcut kick-in distance
    {
      fcut = 1.0;
      fcutderiv = 0.0;
    }
    else if (dx > outer_cutoff)        // Case 2: Our pair distance is greater than the cutoff
    {
      fcut = 0.0;
      fcutderiv = 0.0;
    }
    else                // Case 3: We'll use our modified sin function
    {
      fcut0 = (dx-THRESH) / (outer_cutoff-THRESH) * pi + pi/2.0;
      fcut0_deriv = pi / (outer_cutoff - THRESH);

      fcut = 0.5 + 0.5 * sin(fcut0);
      fcutderiv  = 0.5 * cos(fcut0) * fcut0_deriv;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::get_penalty(const KK_FLOAT dx, const int& pair_idx, KK_FLOAT& E_penalty, KK_FLOAT& force_scalar) const
{
  KK_FLOAT r_penalty = 0.0;

  E_penalty    = 0.0;
  force_scalar = 1.0;

  if (dx - d_penalty_params[0] < c_chimes_2b_cutoff(pair_idx,0)) { // Then we're within the penalty-enforced region of distance space
    r_penalty = c_chimes_2b_cutoff(pair_idx,0) + d_penalty_params[0] - dx;

    /*if (dx < c_chimes_2b_cutoff(pair_idx,0))
      badness = 2;
    else if (1 > this->badness) // Only update badness if candiate badness is worse than its current value
      badness = 1;*/
  }

  if (r_penalty > 0.0) {
    E_penalty    = r_penalty * r_penalty * r_penalty * d_penalty_params[1];

    force_scalar = -3.0 * r_penalty * r_penalty * d_penalty_params[1];

    //if (rank == 0) // Commenting out - we need all ranks to report if the penalty function has been sampled
    //{
        /*cout << "chimesFFKokkos: " << "Adding penalty in 2B Cheby calc, r < rmin+penalty_dist " << fixed
             << dx << " "
             << d_chimes_2b_cutoff(pair_idx,0) + d_penalty_params[0]
             << " pair type: " << pair_idx << endl;
        cout << "chimesFFKokkos: " << "\t...Penalty potential = "<< E_penalty << endl;*/
    //}
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::set_cheby_polys(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                                     const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const
{
  // Currently assumes a Morse-style transformation has been requested

  // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
  // with respect to the interatomic distance, not the transformed distance (x).

  // Do the Morse transformation

  const KK_FLOAT x_min = exp(-1*inner_cutoff/morse);
  const KK_FLOAT x_max = exp(-1*outer_cutoff/morse);

  const KK_FLOAT x_avg   = 0.5 * (x_max + x_min);
  KK_FLOAT x_diff  = 0.5 * (x_max - x_min);

  x_diff *= -1.0; // Special for Morse style

  bool out_of_range;
  const KK_FLOAT dx_orig = dx;

  // The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
  //  function will be zero for dx > outer_cutoff

  if (dx < inner_cutoff) {
    out_of_range = true;
    dx = inner_cutoff;
  } else
    out_of_range = false;

  const KK_FLOAT exprlen = exp(-1*dx/morse);
  const  KK_FLOAT x = (exprlen - x_avg)/x_diff;
  const KK_FLOAT dx_dr = (-exprlen/morse)/x_diff;

  if (!out_of_range) {

    // Generate Chebyshev polynomials by recursion.
    //
    // What we're doing here. Want to fit using Cheby polynomials of the 1st kinD[i]. "T_n(x)."
    // We need to calculate the derivative of these polynomials.
    // Derivatives are defined through use of Cheby polynomials of the 2nd kind "U_n(x)", as:
    //
    // d/dx[ T_n(x) = n * U_n-1(x)]
    //
    // So we need to first set up the 1st-kind polynomials ("Tn[]")
    // Then, to compute the derivatives ("Tnd[]"), first set equal to the 2nd-kind, then multiply by n to get the der's

    // First two 1st-kind Chebys:

    Tn[0] = 1.0;
    Tn[1] = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    Tnd[0] = 1.0;
    Tnd[1] = 2.0 * x;

    // Use recursion to set up the higher n-value Tn and Tnd's

    for (int i = 2; i <= order; i++) {
      Tn[i]  = 2.0 * x *  Tn[i-1] -  Tn[i-2];
      Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }

    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    // The following dx_dr compuation assumes a Morse transformation
    // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

    for (int i = order; i >= 1; i--)
      Tnd[i] = i * dx_dr * Tnd[i-1];

    Tnd[0] = 0.0;
  } else { // out_of_range == true
    //////cout << "Warning: An intermolecular distance less than the inner cutoff = " << inner_cutoff << " was found\n ";
    //////cout << "         Distance = " << dx_orig << endl;

    set_polys_out_of_range(Tn, Tnd, dx_orig, x, order, inner_cutoff, exprlen, dx_dr);
  }
}

#include "chimesFF_kokkos_impl.h"
#endif