# Neural Netwoork Architecture Replication

## Replication Specifications

### Table F.3.2

neural_net_rep.py takes the deeplob architecture that is built from deeplob_rep_inference repo and re-derived layer by layer against the table in Juan's dissertation's (appendix F.3.2).

### Data-Pipeline mismatch

My replication's DataLoader already unsqueezes the data as (N, 1, 100, 40) so if I took DeepLOBDataset. getitem and fed it to my data, it would produce a 5d (N, 1, 1, 100, 40) tensor. 

To fix this, my forward() starts straight at self.conv1(x) so that there isn't an unsqueeze call at all. 

### Layer By Layer re-derivation from table F.2 of Juan's Dissertation

- **conv1** is a (1,2) filter that goes across the time axis and slides across every other entry so that the price and volume stay together. Then two (4,1) convs each shrink the height axis by 3 so it ends up as (N, 32, 94, 20)

- **conv2** follows the same pattern once more taking it to (N,32,88,)

- **conv3** is a single (1,10) filter that collapses the last 10 price levels to 1 in a single step. It ends up being (N,32,82,1)

- **Inception Branches:** all three use padding = 'same' so that none of the change height or width. The only thing that changes is the channel count which goes from 32 -> 64 in every branch