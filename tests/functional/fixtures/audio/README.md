# English speech fixture

Gufo's maintainer supplied this English recording and authorized its
publication (it comes from [Gufo](https://github.com/gufo-org/gufo)'s
`tests/functional/fixtures/audio`) for ASR and TTS voice-cloning functional tests. The sibling text is
the reviewed expected transcript. No additional recording is needed: ASR
concurrency pairs the speech with silence generated in memory.

- WAV SHA-256: `41ff28fbea935f45dae8a760100609bf39a64dccbedc742edb4f0711d516ef5a`
- Mono PCM16, 16 kHz, 10 seconds.

Tests never download voice recordings or depend on a mounted audio folder.
