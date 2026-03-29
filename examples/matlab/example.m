% examples/matlab/example.m
%
% Demonstrates calling the Puddin C API from MATLAB via calllib / loadlibrary.
%
% Prerequisites:
%   1. Build the shared library from the repo root:
%        cargo build --release -p puddin-julia
%   2. Run this script from the repo root, or update REPO_ROOT below.
%
% MATLAB version: R2019b or newer (calllib with pointer aliasing).

% ── locate the library and header ────────────────────────────────────────────
repo_root   = fileparts(fileparts(fileparts(mfilename('fullpath'))));
lib_dir     = fullfile(repo_root, 'target', 'release');
header_file = fullfile(repo_root, 'bindings', 'julia', 'include', 'puddin.h');

if ispc
    lib_file = fullfile(lib_dir, 'puddin_julia.dll');
elseif ismac
    lib_file = fullfile(lib_dir, 'libpuddin_julia.dylib');
else
    lib_file = fullfile(lib_dir, 'libpuddin_julia.so');
end

if ~libisloaded('puddin')
    loadlibrary(lib_file, header_file, 'alias', 'puddin');
end

% ── constants ─────────────────────────────────────────────────────────────────
MSUN = 1.988416e30;

m1 = 30.0 * MSUN;
m2 = 30.0 * MSUN;

fprintf('=== GW150914-like binary (30+30 Msun) ===\n');
fprintf('Total mass      : %.4f Msun\n', calllib('puddin', 'puddin_total_mass', m1, m2) / MSUN);
fprintf('Mass ratio      : %.4f\n',       calllib('puddin', 'puddin_mass_ratio', m1, m2));
fprintf('Sym. mass ratio : %.4f\n',       calllib('puddin', 'puddin_symmetric_mass_ratio', m1, m2));
fprintf('Chirp mass      : %.4f Msun\n',  calllib('puddin', 'puddin_chirp_mass', m1, m2) / MSUN);

a1    = 0.3;   a2    = 0.2;
tilt1 = 0.5236; tilt2 = 1.0472;   % 30 deg, 60 deg in radians

fprintf('chi_eff         : %.4f\n', calllib('puddin', 'puddin_chi_eff', m1, m2, a1, a2, tilt1, tilt2));
fprintf('chi_p           : %.4f\n', calllib('puddin', 'puddin_chi_p',   m1, m2, a1, a2, tilt1, tilt2));

% ── vectorised example using arrayfun ─────────────────────────────────────────
masses_sun = 10:10:50;  % [10 20 30 40 50] Msun equal-mass binaries
mc_sun = arrayfun(@(m) calllib('puddin', 'puddin_chirp_mass', m*MSUN, m*MSUN) / MSUN, masses_sun);

fprintf('\n=== Chirp masses for equal-mass binaries ===\n');
fprintf('  m1=m2 (Msun)  Mc (Msun)\n');
fprintf('  %10.1f  %9.4f\n', [masses_sun; mc_sun]);

unloadlibrary('puddin');
