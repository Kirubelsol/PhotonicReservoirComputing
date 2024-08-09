# PhotonicReservoirComputing
Generating and Analysing Signals for Photonic Reservoir Computing

This repo includes the `python` codes used to generate the input signals (including the preprocessing step) and to train the signal collected from the oscilloscope which is the output of the photonic reservoir computer. These include:

- **Wave generation**: including the sample and hold operation, generating the signal with the specific sampling rate of the AWG, and the masking process specified by the number of virtual nodes. The generated signals include for non-linear channel equalization and signal classification.
- **Scope file edit**: used to filter out and combine the signals collected from the output of the reservoir (photodetector) for training and testing the signal.
- **Numerical Simulation**: to test the equation describing the reservoir and test the performance on different tasks. These are numerical tests, but tests using the actual optoelectronic component in Interconnect and the lab provide the real results.

<div align="center">
    <img src="./img/labsetup.jpg" alt="Experiment Setup" width="300"/> <br>
    Photonic Reservoir Experiment Setup 
</div>