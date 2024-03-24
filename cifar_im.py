def viz_im(model, p, device):
   """
      Creates and returns 2 lists of images from a testloader: one containing real images, the other containing model predictions.

      Parameters
      ----------
      model : nn.Module
         The trained model.   
      p : int
         Number of residual blocks.
      device : string
         The device to use.
      
      Returns
      -------
      list1 : tensor
         List with the true values of each pixel for 4 random testloader images.
      list2 : tensor
         List with predictions for each pixel in 4 random testloader images.
   """
   #global trainloader, testloader, trainset, testset, mean, std, dataset
   
   denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])

   list1 = []
   list2 = []
   rand = torch.randint(0, 625, (4,))

   for i in rand: 
      image, _ = testset[i]

      #Real image
      # Génération de l'image
      
      image_true = (denorm(image)*255).to(torch.int)#Pixel values are normalized, so we need to denormalize them to obtain the original values (0-255)
      image_true = image_true.permute(1, 2, 0)
      list1.append(image_true)

      #Prediction
      y = model(image.to(device), p)
      y = torch.reshape(y, (256, 3, 1024)) #reshape to get a 1d vector, but it still has the 256 channels

      yr_pred = np.zeros(1024) #Channels RGB
      yg_pred = np.zeros(1024)
      yb_pred = np.zeros(1024)

      for i in range(1024):
        probs_r = y[:, 0, i]
        yr_pred[i] = torch.multinomial(probs_r, 1)
        probs_g = y[:, 1, i]
        yg_pred[i] = torch.multinomial(probs_g, 1)
        probs_b = y[:, 2, i]
        yb_pred[i] = torch.multinomial(probs_b, 1)
        
      y_pred = torch.stack([torch.tensor(yr_pred),torch.tensor(yg_pred) ,torch.tensor(yb_pred)]).to(torch.int)
   
      image_hat = torch.reshape(y_pred, (3, 32, 32))
      image_hat = image_hat.permute(1, 2, 0)
      list2.append(image_hat)

   return(list1, list2)



def display_images_in_line(image_list, title):
   """
      Displays images from a list.

      Parameters
      ----------
      image_list : list
         A list of images to view.
      title : string
         The title of the image to view.
   """
   fig = plt.figure(figsize=(len(image_list)*2, 2))
   for i, image_tensor in enumerate(image_list):
      plt.subplot(1, len(image_list), i+1)
      plt.imshow(image_tensor, cmap = 'gray')
      plt.title(title)
      plt.axis('off')
   plt.show()
