# Contributing

Thank you for helping with rusty-gufo. Start with the
[contributor guide](docs/development/README.md): it covers the prerequisites,
the repository layout, the rules every change follows (above all, inference
stays equal to each model's C++ reference) and how to test.

- **Issues:** describe the model and file, the command or request, what you
  expected and what happened; `gufo diagnose` prints the system details that
  help.
- **Pull requests:** keep them focused, follow Conventional Commits in the
  title (`fix(server): …`), and say which checks ran
  ([choosing checks](docs/development/TESTING.md#choosing-checks)). Hosted CI
  runs only the checks that need no GPU; a maintainer runs the full gate on a
  Strix Halo before merging.
- **Coding agents:** [AGENTS.md](AGENTS.md) holds their instructions.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in rusty-gufo by you, as defined in the Apache-2.0
license, shall be dual licensed under the [MIT license](LICENSE-MIT) and the
[Apache License, Version 2.0](LICENSE-APACHE), without any additional terms
or conditions.
