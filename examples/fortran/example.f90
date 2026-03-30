! examples/fortran/example.f90
!
! Demonstrates calling the Puddin C API from modern Fortran, using the
! domain-organised puddin_binary module.
!
! Build (from repo root):
!   cargo build --release -p puddin-julia
!   gfortran examples/fortran/example.f90 \
!       bindings/julia/include/puddin_binary.f90 \
!       -L target/release -lpuddin_julia \
!       -Wl,-rpath,$(pwd)/target/release \
!       -o examples/fortran/example
!   ./examples/fortran/example

program example
    use puddin_binary
    implicit none

    real(c_double) :: m1, m2

    m1 = 30.0_c_double * puddin_binary_MSUN
    m2 = 30.0_c_double * puddin_binary_MSUN

    write(*,'(A)') "=== GW150914-like binary (30+30 Msun) ==="
    write(*,'(A,F8.4,A)') "Total mass      : ", &
        total_mass(m1, m2) / puddin_binary_MSUN, " Msun"
    write(*,'(A,F8.4)')   "Mass ratio      : ", mass_ratio(m1, m2)
    write(*,'(A,F8.4)')   "Sym. mass ratio : ", symmetric_mass_ratio(m1, m2)
    write(*,'(A,F8.4,A)') "Chirp mass      : ", &
        chirp_mass(m1, m2) / puddin_binary_MSUN, " Msun"

    ! Inverse transforms — recover component masses from (Mc, q) and (Mc, eta)
    block
        real(c_double) :: mc, q, eta
        mc  = chirp_mass(m1, m2)
        q   = mass_ratio(m1, m2)
        eta = symmetric_mass_ratio(m1, m2)
        write(*,'(A)') ""
        write(*,'(A)') "=== Inverse transforms ==="
        write(*,'(A,F8.4,A)') "m1 from (Mc, q)   : ", &
            m1_from_chirp_mass_q(mc, q)    / puddin_binary_MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m2 from (Mc, q)   : ", &
            m2_from_chirp_mass_q(mc, q)    / puddin_binary_MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m1 from (Mc, eta) : ", &
            m1_from_chirp_mass_eta(mc, eta) / puddin_binary_MSUN, " Msun"
        write(*,'(A,F8.4,A)') "m2 from (Mc, eta) : ", &
            m2_from_chirp_mass_eta(mc, eta) / puddin_binary_MSUN, " Msun"
    end block

    write(*,'(A)') ""
    write(*,'(A,F8.4)')   "chi_eff         : ", &
        chi_eff(m1, m2, 0.3_c_double, 0.2_c_double, &
                        0.5236_c_double, 1.0472_c_double)
    write(*,'(A,F8.4)')   "chi_p           : ", &
        chi_p(m1, m2, 0.3_c_double, 0.2_c_double, &
                      0.5236_c_double, 1.0472_c_double)

end program example
