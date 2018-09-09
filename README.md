# Dense Neural Network From Scratch

## Step by step naive implementation of densely connected neural network:
- [x] forward operations
- [x] load params from files
- [x] gradient dsec
- [x] params update
- [x] batch processing
- [x] accuracy
- [x] momentum gradient
- [x] 14_100_40_4 network
- [x] 14_28x6_4 network
- [x] 14_14x28_4 network
- [x] plotting
- [ ] gradient checking

## tips for muggle like me:
1. be aware the risk of variable mutation
2. start coding from backprops, use `given_params` to test backprop functions
3. do dimension check with `assert` if not sure
4. write down dimension transformation (consider batch, in/out dim) along the forward and backward and you are good to go
5. loss explosion (layers > 6 in this case) solved by special crafted initialization of weights and bias
6. derive the backprops math by hand will help (with batch, layer in/out dimensions in mind)
