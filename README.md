# Deep Learning Assignement :art:

## Structure of the repository :pushpin:

The repository is composed of two parts : 

- DB/DB
- ClovaAI

### DB :zap:

Here you'll find the main notebook we used for the study : Receipt2BB.ipynb

In this notebook, you'll find :

#### The Data Handling of the SROIE dataset that we downloaded here (https://rrc.cvc.uab.es/?ch=13) :globe_with_meridians: 

- Removed images without groundtruth or groudtruth without their images.
- Removed duplicates : there are many duplicates in the dataset with the same filename :  (X11111111.jpg, X11111111(1).jpg, X11111111(2).jpg)
- Creation of two files with the list of training file names and test file names (test_list.txt : Where there will be all the file names of the test set, train_list.txt : Where there will be all the file names of the training set)
- Rename the ground truth namefiles so that we match the training requirements
- Delete the words from the ground truth files (each line should look like this : x1, y1, x2, y2, x3, y3, x4, y4)

https://static.e-olymp.com/content/98/98c8b20bea775e0a41202ee25564d6f1e777daf2.jpg


#### Training :recycle:

 - Wrote the sroie_resnet18_deform_thre.yaml file in experiments
 
#### Text detection over all test images of SROIE :package:

- Let's now infer all SROIE test images

#### Cropping of each bounding box :pencil2:

- To be clear, imagine for the image 151515.jpg, 300 boxes were infered with the text detector (differentiable binarization)
- We create 300 little pictures of the cropped images that we put in the folder '/home/jupyter/Clova/demo_image/' 

```python
image_folder = "/home/jupyter/DB/DB/datasets/SROIE/test_images/"

for file in filenames:
    BB = pd.read_fwf('/home/jupyter/Clova/BB_infered_with_DB/'+file+'.txt', header = None) # List of all text bounding boxes of a file infered by DB

    #Extract the k eme bounding box
    path = '/home/jupyter/Clova/demo_image/'+file
    os.mkdir(path)
    for k in range(BB.shape[0]):
        #Read the image without bounding boxes
        img = cv2.imread(image_folder+file[4:]+'.jpg')
        l_char = BB[0][k].split(",")
        l = [int(i) for i in l_char[:8]] # [x1, y1, x2, y2, x3, y3, x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]

        crop_img = img[y1:y3, x1:x2]
        
        filename = '/'+file+'_demo_'+str(k)+'.jpg'
        
        # Using cv2.imwrite() method 
        # Saving the image 
        try:
            cv2.imwrite(path + filename, crop_img)
        except:
            continue
```

- We finally get 360 folders full of cropped images for the 360 test images 

### Clova :memo:

#### Text recognition over the cropped images :card_file_box:

- In the folder ClovaAI, you'll find a script I wrote to infer text over each bounding box and put it in a text file (Clova/BB2text.py)

- We launch the file BB2text.py that I wrote in order to write down the predictions the way we want


- Here's the format that we get : ("x1, y1, x2, y2, x3, y3, x4, y4, detection_probability, word, recognition_probability")

#### From words to sentences :speech_balloon:

- We used a "nearest neighbor" methodology in order to put together the words that we estimate from the same sentence 

This means that we have to the following steps :
- measure the maximal vertical size of the word : max(y3-y1)
- Find the closest word of word 1
- if right distance is inferior to the vertical height multiplied by 1.5 : add it to the right of the word
- if left distance is inferior to the vertical height multiplied by 1.5 : add it to the left of the word
