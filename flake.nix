{
  description = "gufo - Gufo Engine for AMD Strix Halo (gfx1151 GPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Reference ASR/TTS runtime for benchmark comparisons; exposes a
    # ROCm gfx1151 package tuned for Strix Halo.
    audio-cpp = {
      url = "github:0xShug0/audio.cpp";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, audio-cpp }:
    let
      # Strix Halo is a Linux x86-64-only target (see README non-goals).
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" ];
      pkgs = forAllSystems (system: import nixpkgs { inherit system; });

      llamaReferences = forAllSystems (
        system: pkgs.${system}.callPackage ./.devops/nix/llama-cpp-reference.nix { }
      );

      # Model reference and client-validation tools; never a transitive
      # dependency of the server (see tools/README.md for the offline toolchain).
      # Unified python313 + torchWithRocm: every Strix Halo box ships ROCm, so
      # the single toolchain serves CPU flows and the --device cuda
      # reference forward alike. gfx1151 verified on this host.
      pythonTools = system:
        let pt = pkgs.${system}.python313; in
        pt.withPackages (
          ps:
          let
            # Nixpkgs' default torchvision closure uses CPU torch; override
            # it so evaluation has one ROCm torch derivation.
            torchvisionRocm = ps.torchvision.override {
              torch = ps.torchWithRocm;
            };
            accelerateRocm = ps.accelerate.override {
              torch = ps.torchWithRocm;
            };
          in
          [
            ps.torchWithRocm
            torchvisionRocm
            ps.transformers
            accelerateRocm
            ps.safetensors
            ps.huggingface-hub
            ps.requests
            ps.openai # official SDK for local API compatibility checks
            ps.jsonschema # independent structured-output schema oracle
            ps.numpy
            ps.scipy
            ps.matplotlib # benchmark charts (tools/bench/model-bench.py render)
          ]
        );
    in
    {
      # Gufo itself builds with Cargo against ROCm 10 (README); nixpkgs ships
      # ROCm 7.2.3, so no package of it is defined here yet. These are the
      # optional benchmark references.
      packages = forAllSystems (
        system:
        {
          ds4-reference = pkgs.${system}.callPackage ./.devops/nix/ds4-reference.nix { };
          llama-cpp-reference = llamaReferences.${system}.release;
          llama-cpp-mtp-reference = llamaReferences.${system}.mtp;
        }
      );

      devShells = forAllSystems (
        system:
        {
          # Unified toolchain (CPU + ROCm torch): one default shell serves all
          # offline CPU flows and the ROCm reference forward.
          default = pkgs.${system}.mkShell {
            packages = [
              (pythonTools system)
              pkgs.${system}.clang-tools
              pkgs.${system}.ffmpeg-headless
              pkgs.${system}.sox
              pkgs.${system}.rocmPackages.rocprofiler-sdk
              pkgs.${system}.sqlite
              audio-cpp.packages.${system}.rocm-gfx1151
            ];
            env = {
              LD_LIBRARY_PATH = pkgs.${system}.lib.makeLibraryPath [
                pkgs.${system}.stdenv.cc.cc.lib
                pkgs.${system}.zlib
              ];
            };
          };

        }
      );
    };
}
