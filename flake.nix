{
  description = "LTR_Rubric";

  # Inputs
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";
  inputs.exampp.url = "git+https://github.com/laura-dietz/rubric-internal";

  # Outputs
  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, exampp, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Create a Python environment for CUDA
        mkShell = target: (dspy-nix.lib.${system}.mkShell {
          inherit target;
          pythonOverrides = [ exampp.lib.${system}.pythonOverrides ];
          pythonPackages = ps: [ ps.exampp ps.scikit-learn ps.mypy ps.pylatex ];
        });

        # Define Python package overrides for your project
        pythonOverrides = pkgs.lib.composeOverlays
          exampp.lib.${system}.pythonOverrides
          (self: super: {
            rubric_llm_judge = self.buildPythonPackage {
              name = "rubric_llm_judge";
              src = ./.;
              format = "pyproject";
              propagatedBuildInputs = with self; [
                setuptools
                pydantic
                exampp
              ];
            };
          });

      in {
        # Expose Python overrides for the system
        lib.pythonOverrides = pythonOverrides;

        # Define the CUDA-specific development shell
        devShells.default = mkShell "cuda";
      }
    );
}
