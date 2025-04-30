# DFL Study Based on DL

This repository contains the codebase developed for my Master's degree thesis. The abstract is the following:

Decentralized Federated Learning (DFL) presents a promising paradigm for collaborative machine learning without the need for centralized data aggregation, addressing growing concerns over data privacy and regulatory compliance. This study proposes and evaluates a DFL system specifically designed for audio-to-text conversion, employing a convolutional neural network as the learning model. The system integrates the Ethereum blockchain for consensus and the InterPlanetary File System for decentralized model storage, enabling deployment in environments where participants cannot be fully managed, trusted, or predetermined. The evaluation assessed not only the system's performance under safeguards against inference attacks but also its resilience in scenarios involving data imbalance across nodes and poisoning attacks. Results indicate that the DFL system maintained competitive performance compared to centralized counterparts under most conditions and demonstrated substantial resilience to both data imbalance and poisoning. However, significant performance degradation was observed when high levels of noise were introduced to enhance protection against inference attacks. Overall, this work contributes a practical implementation of a privacy-preserving, decentralized learning system for audio-to-text conversion, laying a foundation for future research and applications in secure, user-driven, and regulation-compliant distributed machine learning environments.

The repository is organized into three main parts:

* experiments/ — Code used to perform the experiments presented in the thesis report.

* dfl_system/ — Core implementation of the DFL system.

* report/ - The thesis full report.

Thesis Reference
If you use this code or ideas from it, please cite or reference the thesis:

Tiago Ferreira, "DFL Study Based on DL," Master's thesis, ISEC, 2025.

License
This code is released under the Apache 2.0 License. See the LICENSE file for more details.
