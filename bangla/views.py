from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from .forms import ImageForm
from PIL import Image
from googletrans import Translator, constants
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel



def show_generate(url, greedy = True):
    translator = Translator()
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    image = Image.open(url)
    pixel_values   = image_processor(image, return_tensors ="pt").pixel_values

    if greedy:
        generated_ids  = model.generate(pixel_values, max_new_tokens = 30)
    else:
        generated_ids  = model.generate(
            pixel_values,
            do_sample=True,
            max_new_tokens = 30,
            top_k=5)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    bangla = translator.translate(generated_text, dest='bn')
    return bangla.text


def image_upload_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            # url = 'F:\project' + img_obj.image.url
            url = img_obj.image.url
            text_img = show_generate(url, greedy = False)
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj, 'caption': text_img})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})
