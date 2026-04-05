import google.generativeai as genai


genai.configure(api_key='AIzaSyB2wstzxTbCuiTF5UqdQ4xiX-gWgVlcrWY')
for m in genai.list_models():
      if 'embed' in m.name:
          print(m.name, m.supported_generation_methods)