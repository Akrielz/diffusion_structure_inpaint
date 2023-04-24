#include <pybind11/pybind11.h>

int main(int argc, char** argv);
int faspr_run(string pdbin, string pdbout, string seqfile, bool sflag);

namespace py = pybind11;

PYBIND11_MODULE(faspr, mod) {
	mod.def("FASPR_cpp", &faspr_run, "FASPR side-chain modeling.");
}
