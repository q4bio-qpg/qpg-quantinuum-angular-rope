# Tests of the Angular/RoPE quantum DNA encoding on Quantinuum's hardware (H2 and Helios systems)

In these tests, 16 pairs of DNA 100,000-mers where randomly generated with varying Levenshtein distances from the list [199, 651, 1321, 2133, 3139, 4253, 5427, 6729, 8154, 9486, 11025, 12721, 14931, 15991, 17930, 19634]. For each DNA kmer, a RoPE encoding on 9 qubits (512 complex dimensions) was constructed. Then, the encodings were transformed into quantum circuits using the Angular encoding targeting various numbers of qubits in different tests (e.g. 20, 56, 98). In each pair, the fidelity between encodings was estimated by applying one circuit to the zero state, then the conjugate of the other circuit, and finally measuring the frequency of the zero state (this is known as the Inversion Test or Overlap Test). See the cited paper for more details. 

### Description of the files: 

- gen_test_data.jl, utils.jl - the Julia code used to generate kmer pairs and compute their RoPE encodings
- ropes_v2.npy - the computed RoPE encodings for 16 pairs of generated kmers
- points_v2.npy - the Levenshtein distances and fidelities between RoPE encodings for the 16 pairs
- anszats.py - code for the Angular encoding (standard and compact versions)
- helios_leakage.ipynb - executing the Overlap Tests on Quantinuum's Helios systems with leakage mesurements
- h2_compact.ipynb - executing the Overlap Tests on Quantinuum's H2 systems using the compact Angular encoding

# Citation

```
@misc{yakymenko2026rotormapquantumfingerprintsdna,
      title={RotorMap and Quantum Fingerprints of DNA Sequences via Rotary Position Embeddings}, 
      author={Danylo Yakymenko and Maksym Chernyshev and Illia Savchenko and Sergii Strelchuk},
      year={2026},
      eprint={2603.22245},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.22245}, 
}
```
