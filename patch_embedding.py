import torch
import numpy as np
import torch.nn as nn 

trainData = np.loadtxt("dataset/ECG5000/ECG5000_TRAIN.txt")
testData = np.loadtxt("dataset/ECG5000/ECG5000_TEST.txt")

data = np.vstack([trainData, testData])

labels = data[:, 0].astype(int)
signals = data[:,1: ]

normal_signals = signals[labels == 1]
x = torch.tensor(normal_signals, dtype=torch.float32)

#-- patch embedding --
patchSize = 10
batchSize, signal_length = x.shape

assert signal_length % patchSize == 0

patchNum = signal_length // patchSize

x_patches = x.view(batchSize, patchNum, patchSize)
print("Patches shape:", x_patches.shape) 

#--- vector representation ---

#representing each patch as a vector
dModel = 64

embeddedPatches = nn.Linear(10, dModel)
xEmbedded = embeddedPatches(x_patches)

print("Embedded patches shape:", xEmbedded.shape)

#--- positional encoding ---

batchSize, patchNum, dModel = xEmbedded.shape

positionalEmbedding = nn.Parameter(torch.zeros(1, patchNum, dModel))

xWithPositions = positionalEmbedding + xEmbedded

print ("with positions:", xWithPositions.shape)

#--- transformer encoder ---

numHeads = 4
numLayers = 2
dimFeedforward = 128
dropout = 0.1

encoderLayer = nn.TransformerEncoderLayer(
    dModel,          # 64
    nhead=numHeads,       
    dim_feedforward=dimFeedforward,
    dropout=dropout,
    batch_first=True           # input is [batch, seq, features]
)

transformerEncoder = nn.TransformerEncoder(
    encoderLayer,
    num_layers=numLayers
)

xEncoded = transformerEncoder(xWithPositions)

print("Encoded output shape:", xEncoded.shape)

#--- transformer decoder ---

decoder = nn.Linear(dModel, patchSize)
xDecodedPatches = decoder(xEncoded)

print("Decoded patches shape:", xDecodedPatches.shape)

#--- reconstruct full signal ---

xReconstructed = xDecodedPatches.contiguous().view(batchSize, signal_length)

print("Reconstructed signal shape:", xReconstructed.shape)

#--- reconstruction loss ---

lossFn = nn.MSELoss()
loss = lossFn(xReconstructed, x)

print("Reconstruction loss:", loss.item())
