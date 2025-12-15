## A modified recipe for robust SVS. The original recipe is build on top of VISinger2+Opencpop

🎙️ Task: Singing Voice Synthesis (SVS)

📊 Corpus: Mandarin Single Female Singer Corpus, Opencpop.

Before start working on this recipe, please refer to tutorials

* [PSC usage tutorial](https://www.wavlab.org/activities/2022/psc-usage/)
* [Espnet recipe tutorial](https://github.com/espnet/notebook/blob/master/ESPnet2/Course/CMU_SpeechRecognition_Fall2022/recipe_tutorial.ipynb)


Set using posterior uncertainty predictor at ``./espnet2/gan_svs/vits/vits.py``

Set using prior uncertainty feature augmentation at ``./espnet2/gan_tts/hifigan/loss.py``, ``Diffaug`` entry. The same type of differentiable augmentation should be used in both the generator and the discriminator.

