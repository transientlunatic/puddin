! examples/fortran/example.f90
!
! Demonstrates calling the Puddin C API from modern Fortran via iso_c_binding.
! No link-time tricks are needed; Fortran's iso_c_binding maps directly onto
! the C ABI symbols from libpuddin_julia.
!
! Build (from repo root):
!   cargo build --release -p puddin-julia
!   gfortran examples/fortran/example.f90 \
!       -L target/release -lpuddin_julia \
!       -Wl,-rpath,$(pwd)/target/release \
!       -o examples/fortran/example
!   ./examples/fortran/example

program example
    use iso_c_binding, only: c_double
    implicit none

    ! ── Declare the C interface ───────────────────────────────────────────────
    interface
        function puddin_total_mass(m1, m2) bind(C, name="puddin_total_mass")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: puddin_total_mass
        end function

        function puddin_mass_ratio(m1, m2) bind(C, name="puddin_mass_ratio")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: puddin_mass_ratio
        end function

        function puddin_symmetric_mass_ratio(m1, m2) &
                bind(C, name="puddin_symmetric_mass_ratio")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: puddin_symmetric_mass_ratio
        end function

        function puddin_chirp_mass(m1, m2) bind(C, name="puddin_chirp_mass")
            import c_double
            real(c_double), value :: m1, m2
            real(c_double)        :: puddin_chirp_mass
        end function

        function puddin_m1_from_mc_q(mc, q) bind(C, name="puddin_m1_from_mc_q")
            import c_double
            real(c_double), value :: mc, q
            real(c_double)        :: puddin_m1_from_mc_q
        end function

        function puddin_m2_from_mc_q(mc, q) bind(C, name="puddin_m2_from_mc_q")
            import c_double
            real(c_double), value :: mc, q
            real(c_double)        :: puddin_m2_from_mc_q
        end function

        function puddin_m1_from_mc_eta(mc, eta) bind(C, name="puddin_m1_from_mc_eta")
            import c_double
            real(c_double), value :: mc, eta
            real(c_double)        :: puddin_m1_from_mc_eta
        end function

        function puddin_m2_from_mc_eta(mc, eta) bind(C, name="puddin_m2_from_mc_eta")
            import c_double
            real(c_double), value :: mc, eta
            real(c_double)        :: puddin_m2_from_mc_eta
        end function

        function puddin_chi_eff(m1, m2, a1, a2, tilt1, tilt2) &
                bind(C, name="puddin_chi_eff")
            import c_double
            real(c_double), value :: m1, m2, a1, a2, tilt1, tilt2
            real(c_double)        :: puddin_chi_eff
        end function

        function puddin_chi_p(m1, m2, a1, a2, tilt1, tilt2) &
                bind(C, name="puddin_chi_p")
            import c_double
            real(c_double), value :: m1, m2, a1, a2, tilt1, tilt2
            real(c_double)        :: puddin_chi_p
        end function
    end interface

    real(c_double), parameter :: MSUN = 1.988416e30_c_double
    real(c_double)            :: m1, m2

    m1 = 30.0_c_double * MSUN
    m2 = 30.0_c_double * MSUN

    write(*,'(A)') "=== GW150914-like binary (30+30 Msun) ==="
    write(*,'(A,F8.4,A)') "Total mass      : ", &
        puddin_total_mass(m1, m2) / MSUN, " Msun"
    write(*,'(A,F8.4)')   "Mass ratio      : ", puddin_mass_ratio(m1, m2)
    write(*,'(A,F8.4)')   "Sym. mass ratio : ", puddin_symmetric_mass_ratio(m1, m2)
    write(*,'(A,F8.4,A)') "Chirp mass      : ", &
        puddin_chirp_mass(m1, m2) / MSUN, " Msun"

    ! Inverse transforms — recover component masses from (Mc, q) and (Mc, eta)
    block
        real(c_double) :: mc, q, eta
        mc  = puddin_chirp_mass(m1, m2)
        q   = puddin_mass_ratio(m1, m2)
        eta = puddin_symmetric_mass_ratio(m1, m2)
        write(*,'(A)') ""
        write(*,'(A)') "=== Inverse transforms ==="
        write(*,'(A,F8.4,A)') "m1 from (Mc, q)   : ", &
            puddin_m1_from_mc_q(mc, q)    / MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m2 from (Mc, q)   : ", &
            puddin_m2_from_mc_q(mc, q)    / MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m1 from (Mc, eta) : ", &
            puddin_m1_from_mc_eta(mc, eta) / MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m2 from (Mc, eta) : ", &
            puddin_m2_from_mc_eta(mc, eta) / MSUN, " Msun"
    end block

    write(*,'(A)') ""
    write(*,'(A,F8.4)')   "chi_eff         : ", &
        puddin_chi_eff(m1, m2, 0.3_c_double, 0.2_c_double, &
                               0.5236_c_double, 1.0472_c_double)
    write(*,'(A,F8.4)')   "chi_p           : ", &
        puddin_chi_p(m1, m2, 0.3_c_double, 0.2_c_double, &
                             0.5236_c_double, 1.0472_c_double)

end program example
