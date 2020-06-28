## Preprocessing
## Hough
## Canny
## HSV
## Gaussian Blur
## PyTorch
 - Tensor: Was ist das?
## OpenCV
## CSICamera
## Robot
## Nvidia Jetson
 - Speccs
 - GPU und CPU, gegenüberstellung
 - Bla
## Google Colab
## jstest.py
 - Steuerung mit Controller + Aufnahme
## snapshot.py
 - Evaluierung der Bildqualität
## line_detection_from_folder.py
## Allgemeine Herangehensweise bei ML Problemen
 - Daten Daten Daten
 - Richtig Gelabelte Daten (Supervised Learning)
## Anwendugsgebiete ML
 - Werbung
 - Netflix vorschläge
 - Pattern Matching (Medizin)
 - Autonom. Fahren
    * Tesla
## Verschiedene Learning Modelle
 - Supervised (Unser Projekt)
 - Unsupervised
 - Reinforced
## Neuronale Netze Allgemein
 - Was ist ein NN?
 - Layer
 - Neuronen / Perceptron
 - Was passiert beim trainieren?
 - SGD
 - Overfitten/Underfitten/Just Right
 - Learning Rate
 - Weight & Biases (schöne Graphen)
 - Wir benutzen ein CNN (Resnet), was ist das überhaupt?
 - Activation Functions
    - Sigmoid 
    - ReLU
    - Softmax
    - Was ist das überhaupt?
    - Wozu sind die gut?
## ResNet
 - ResNet18, 36...
 - Funktionsweise
 - Architektur
 - PreTrained
 - https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
## Must mention
- Resnet can only work with RGB pictures --> Grayscale images have to be adjusted
## Nice to know


# Ablauf
 - Was ist ein NN?
 - Learning Modelle
 - Wie trainiert man ein (supervised learning) NN?
 - Was gibt es für Modelle (Architekturen ResNet, ImageNet...)
 - Anwendungsgebiete
 - Anwendungsgebiet Autonom. Fahren (überleitung zu unserem Thema Road Detection)
 - Preprocessing 
    - ROI
    - Canny
    - Maske
    - Hough
    - Transformieren
    - Normalisieren
 - Wie trainiert man das Modell mit seinen Daten?
    - Was muss man beim Trainieren beachten?
    - Shuffeln
    - Balanced Dataset
    - Qualität des Datensets
    - Richtige Label
 - Wie verwendet man sein trainiertes Modell?
    - Transfer Learning, ja nein
 - Camera Input
 - Auto lenkt anhand der Infered Bilder
   - Output wird anhand des Inputs richtig vorhergesagt
 - LiveDemo!!!!!!

# Ideen für die Präsi
- Dem Publikum ein paar unserer Bilder zeigen und fragen wie sie die labeln würden -> Richtiges labeln ist wichtig, aber auch schwierig (zumindest für uns)
- Kurzes (evtl. 10s) Video zeigen wie das Auto in der Entwicklung fährt 
  - Mit den 3 label
  - Mit Code aus der BA
  - Vielleicht verbesserungen zeigen
    - "Dann haben wir das gemacht und so hat sichs ausgewirkt" -> Kurzes video
- Was ist anders zwischen Deep learning und ML auf embedded systems?