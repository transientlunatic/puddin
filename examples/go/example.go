// examples/go/example.go
//
// Demonstrates calling the Puddin C API from Go via cgo.
//
// Build (from repo root):
//
//	cargo build --release -p puddin-julia
//	cd examples/go && go run example.go
//
// Or build a binary:
//
//	cd examples/go && go build -o example && ./example

package main

// #cgo CFLAGS:  -I../../bindings/julia/include
// #cgo LDFLAGS: -L../../target/release -lpuddin_julia -Wl,-rpath,../../target/release
// #include "puddin.h"
import "C"

import "fmt"

func main() {
	const msun = C.PUDDIN_MSUN

	m1 := C.double(30.0 * msun)
	m2 := C.double(30.0 * msun)

	fmt.Println("=== GW150914-like binary (30+30 Msun) ===")
	fmt.Printf("Total mass      : %.4f Msun\n", float64(C.puddin_total_mass(m1, m2))/msun)
	fmt.Printf("Mass ratio      : %.4f\n", float64(C.puddin_mass_ratio(m1, m2)))
	fmt.Printf("Sym. mass ratio : %.4f\n", float64(C.puddin_symmetric_mass_ratio(m1, m2)))
	fmt.Printf("Chirp mass      : %.4f Msun\n", float64(C.puddin_chirp_mass(m1, m2))/msun)

	a1 := C.double(0.3)
	a2 := C.double(0.2)
	tilt1 := C.double(0.5236) // 30 deg
	tilt2 := C.double(1.0472) // 60 deg

	fmt.Printf("chi_eff         : %.4f\n", float64(C.puddin_chi_eff(m1, m2, a1, a2, tilt1, tilt2)))
	fmt.Printf("chi_p           : %.4f\n", float64(C.puddin_chi_p(m1, m2, a1, a2, tilt1, tilt2)))
}
