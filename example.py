class MyApp(fa.FastApp):
    def model(
        self,
        embedding_dim:int = fa.Param(default=16, min=4, max=32, help="The size of the embedding for each character."),
        sentence_dim:int =fa.Param(default=128, min=32, max=1500, help="The size of the hidden units in the LSTM sentence processor."),
    ) -> nn.Module:
        # Create pytorch module
        return nn.Sequential(
            ...
        )

    def dataloaders(
        inputs:Path = fa.Param(help="The input file."), 
        batch_size:int = fa.Param(default=32, help="The batch size."),
    ):
        return 