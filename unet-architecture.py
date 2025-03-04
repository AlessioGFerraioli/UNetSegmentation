class DownsamplingLayer(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=5):
      super().__init__()
      self.layer = nn.Sequential(
         nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
         nn.ReLU(),
         nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
         nn.ReLU()
      )   # definizione del layer convoluzionale
      self.pool = nn.MaxPool2d(2, 2)   # operazione di max pooling per dimezzare le dimensioni dell'immagine

   def forward(self, x):
      x = self.layer(x)
      y = self.pool(x)
      return y, x   # restituzione delle feature ridotte e non per permettere le skip connections


class UpsamplingLayer(nn.Module):
   def __init__(self, out_channels, kernel_size=5):
      super().__init__()
      self.layer = nn.Sequential(
         nn.LazyConvTranspose2d(out_channels, 2, 2),
         nn.ReLU(),
         nn.Conv2d(out_channels, out_channels, kernel_size),
         nn.ReLU(),
         nn.Conv2d(out_channels, out_channels, kernel_size),
         nn.ReLU()
      )

   def forward(self, x, s):   # input: feature layer precedente e skip connection
      y = self.layer(torch.cat([x, s], dim=1))
      return y


class UNet(nn.Module):
   def __init__(self, in_channels=3, out_channels=1, image_dim=(256, 256)):
      super().__init__()
      H, W = image_dim 
      
      # Downsampling  
      self.down_layer_1 = DownsamplingLayer(in_channels, 64)   # 64 x H // 2 x W // 2
      self.down_layer_2 = DownsamplingLayer(64, 128)   # 128 x H // 4 x W // 4
      self.down_layer_3 = DownsamplingLayer(128, 256)   # 256 x H // 8 x W // 8
      self.down_layer_4 = DownsamplingLayer(256, 512)   # 256 x H // 16 x W // 16
      
      # Embedding layer
      self.embedding_layer = nn.Sequential(
         nn.Conv2d(512, 1024, 5, padding='same'),
         nn.ReLU(),
         nn.Conv2d(1024, 1024, 5, padding='same'),
         nn.ReLU()
      )   # 1024 x H // 16 x W // 16

      # Upsampling
      self.up_layer_1 = UpsamplingLayer(512)   # 512 x H // 8 x W // 8
      self.up_layer_2 = UpsamplingLayer(256)   # 256 x H // 4 x W // 4
      self.up_layer_3 = UpsamplingLayer(128)   # 128 x H // 2 x W // 2
      self.up_layer_4 = UpsamplingLayer(64)    # 64 x H x W

      # Classificatore
      self.classifier = nn.Sequential(
         nn.Conv2d(64, 1, 1),
         nn.Sigmoid()
      )

   def forward(self, image):
      z, z_1 = self.down_layer_1(image)
      z, z_2 = self.down_layer_2(z)
      z, z_3 = self.down_layer_3(z)
      z, z_4 = self.down_layer_4(z)

      z = self.embedding_layer(z)

      z = self.up_layer_1(z, z_4)
      z = self.up_layer_2(z, z_3)
      z = self.up_layer_3(z, z_2)
      z = self.up_layer_4(z, z_1)

      y = self.classifier(z)
      return y