# torchbci: 

# Build HTML Doc

If you make updates to the source code, you can always re-build HTML doc and even host it locally! We use `google` format for the docstring, so please make sure you are following its format!

1. Install the `pdoc` package via pip:
`pip install pdoc`
2. Just simply run pdoc:
`pdoc ./torchbci -o ./doc --docformat google`

# To-do List:
Feel free to contribute! Make sure you create a new branch and contribute via pull requests.
- [ ] Think of a better name of this framework than `torchbci`
- [ ] Implement `DataStreamer`
- [ ] Finish implementation and demo tutorial for `Jims Algorithm`
- [ ] Add and clean up docstring for existing code
- [ ] Get Kilosort work within this framework
- [ ] Confirm the api protocol between backend and visualization frontend
- [ ] Add useful functions to the functional interface 