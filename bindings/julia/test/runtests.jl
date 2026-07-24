using Test
using Puddin

const MSUN = Puddin.MSUN

@testset "Puddin.jl" begin

    @testset "total_mass" begin
        @test total_mass(30.0 * MSUN, 30.0 * MSUN) ≈ 60.0 * MSUN  rtol=1e-10
        @test total_mass(10.0 * MSUN,  5.0 * MSUN) ≈ 15.0 * MSUN  rtol=1e-10
    end

    @testset "mass_ratio" begin
        @test mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 1.0   rtol=1e-10
        @test mass_ratio(30.0 * MSUN, 10.0 * MSUN) ≈ 1/3   rtol=1e-10
    end

    @testset "symmetric_mass_ratio" begin
        @test symmetric_mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 0.25  rtol=1e-10
        # η ≤ 1/4 for all mass ratios
        @test symmetric_mass_ratio(30.0 * MSUN, 10.0 * MSUN) ≤ 0.25
    end

    @testset "chirp_mass" begin
        # For equal masses m: Mc = 2m * (1/4)^(3/5)
        m = 30.0 * MSUN
        mc_expected = 2.0 * m * 0.25^(3/5)
        @test chirp_mass(m, m) ≈ mc_expected  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_q roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc = chirp_mass(m1, m2)
        q  = mass_ratio(m1, m2)
        (r1, r2) = masses_from_chirp_mass_q(mc, q)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_q equal masses" begin
        m = 30.0 * MSUN
        mc = chirp_mass(m, m)
        (r1, r2) = masses_from_chirp_mass_q(mc, 1.0)
        @test r1 ≈ m  rtol=1e-10
        @test r2 ≈ m  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_eta roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc  = chirp_mass(m1, m2)
        eta = symmetric_mass_ratio(m1, m2)
        (r1, r2) = masses_from_chirp_mass_eta(mc, eta)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "masses_from_chirp_mass_eta equal masses" begin
        m = 30.0 * MSUN
        mc = chirp_mass(m, m)
        (r1, r2) = masses_from_chirp_mass_eta(mc, 0.25)
        @test r1 ≈ m  rtol=1e-10
        @test r2 ≈ m  rtol=1e-10
    end

    @testset "chi_eff — non-spinning" begin
        m = 30.0 * MSUN
        @test chi_eff(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "chi_eff — aligned spins" begin
        m = 30.0 * MSUN
        # Both spins fully aligned: chi_eff = (m1*a1 + m2*a2) / M = 0.5
        @test chi_eff(m, m, 0.5, 0.5, 0.0, 0.0) ≈ 0.5  rtol=1e-10
    end

    @testset "chi_p — non-spinning" begin
        m = 30.0 * MSUN
        @test chi_p(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "broadcasting" begin
        # Julia broadcasting should work transparently over scalar ccall wrappers
        m1 = [30.0, 10.0, 5.0] .* MSUN
        m2 = [30.0,  5.0, 5.0] .* MSUN
        mc = chirp_mass.(m1, m2)
        @test length(mc) == 3
        @test all(mc .> 0.0)
    end

    # ── Puddin.Binary submodule ───────────────────────────────────────────────

    @testset "Binary submodule accessible" begin
        @test isdefined(Puddin, :Binary)
        @test Puddin.Binary isa Module
    end

    @testset "Binary.chirp_mass matches top-level" begin
        m = 30.0 * MSUN
        @test Puddin.Binary.chirp_mass(m, m) ≈ chirp_mass(m, m)  rtol=1e-10
    end

    @testset "Binary.total_mass" begin
        @test Puddin.Binary.total_mass(30.0 * MSUN, 30.0 * MSUN) ≈ 60.0 * MSUN  rtol=1e-10
    end

    @testset "Binary.mass_ratio" begin
        @test Puddin.Binary.mass_ratio(30.0 * MSUN, 15.0 * MSUN) ≈ 0.5  rtol=1e-10
    end

    @testset "Binary.symmetric_mass_ratio" begin
        @test Puddin.Binary.symmetric_mass_ratio(30.0 * MSUN, 30.0 * MSUN) ≈ 0.25  rtol=1e-10
    end

    @testset "Binary.masses_from_chirp_mass_q roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc = Puddin.Binary.chirp_mass(m1, m2)
        q  = Puddin.Binary.mass_ratio(m1, m2)
        (r1, r2) = Puddin.Binary.masses_from_chirp_mass_q(mc, q)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "Binary.masses_from_chirp_mass_eta roundtrip" begin
        m1 = 30.0 * MSUN
        m2 = 20.0 * MSUN
        mc  = Puddin.Binary.chirp_mass(m1, m2)
        eta = Puddin.Binary.symmetric_mass_ratio(m1, m2)
        (r1, r2) = Puddin.Binary.masses_from_chirp_mass_eta(mc, eta)
        @test r1 ≈ m1  rtol=1e-10
        @test r2 ≈ m2  rtol=1e-10
    end

    @testset "Binary.chi_eff" begin
        m = 30.0 * MSUN
        @test Puddin.Binary.chi_eff(m, m, 0.5, 0.5, 0.0, 0.0) ≈ 0.5  rtol=1e-10
    end

    @testset "Binary.chi_p" begin
        m = 30.0 * MSUN
        @test Puddin.Binary.chi_p(m, m, 0.0, 0.0, 0.0, 0.0) ≈ 0.0  atol=1e-15
    end

    @testset "Binary broadcasting" begin
        m1 = [30.0, 10.0] .* MSUN
        m2 = [30.0,  5.0] .* MSUN
        mc = Puddin.Binary.chirp_mass.(m1, m2)
        @test length(mc) == 2
        @test all(mc .> 0.0)
    end

end
