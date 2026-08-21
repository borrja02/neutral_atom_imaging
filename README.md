# High-efficiency fluorescence imaging pipeline for 87Rb neutral-atom experiments

Official repository for the Master's Thesis: **"Imaging protocols for arrays of single neutral atoms"**.

This repository contains the full codebase, Zemax OpticStudio simulation results, photon budget models, and the final thesis manuscript (`memoria.pdf`).

## Overview

Achieving high-fidelity readout in optical tweezer arrays requires maximizing the Signal-to-Noise Ratio (SNR) within short exposure times to minimize atomic heating. This project implements a hybrid computational approach integrating Zemax OpticStudio with a custom Python grid-search algorithm to filter and evaluate thousands of candidate commercial lens configurations based on magnification, track length, pupil filling, and physical optics performance.

## Abstract

This master's thesis presents the design and optimization of a high-efficiency fluorescence imaging pipeline for $^{87}\text{Rb}$ neutral-atom experiments. Achieving high-fidelity readout in optical tweezer arrays requires maximizing the Signal-to-Noise Ratio (SNR) within short exposure times to minimize atomic heating. The methodology employs a hybrid computational approach: Zemax OpticStudio was integrated alongside a custom Python-based grid-search algorithm to enforce proper ray tracing filters and systematically evaluate candidate lens configurations from commercial catalogs, filtering thousands of optical configurations based on magnification, track length, and pupil filling constraints. The resulting best configurations were further modeled in Zemax to perform rigorous physical optics simulations. This analysis reveals that standard spherical optics suffer from severe aberrations, identifying a symmetric aspheric relay (ASP-2) as the optimal architecture for maximizing optical performance across the whole field of view.

Regarding detection, a photon budget was established by modeling trap-induced AC Stark shifts, yielding ~800 incident photons per 10 ms exposure. Comparative modeling demonstrates that quantitative CMOS (qCMOS) technology significantly outperforms CCD and EMCCD sensors. By combining sub-electron read noise with a unity excess noise factor, qCMOS maintains > 95% efficiency relative to the Standard Quantum Limit, ensuring robust state discriminability even under magnification-induced spatial spreading. These findings offer a framework for evaluating readout optics, paving the way toward scalable high-fidelity detection in neutral-atom systems.

