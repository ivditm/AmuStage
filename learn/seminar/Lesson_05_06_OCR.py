import streamlit as st
import easyocr
import cv2
import numpy as np
import base64
import os
from io import StringIO 
from easyocr import Reader

from PIL import Image
from PIL import ImageDraw

from functions import draw_boxes, set_background

IMG_PATH = 'изображения/'


#--------------------------------------------------------
# НЕ забыть сделать виртуальную среду и работать в ней!!!

# в_папке_проекта$ python3 -m venv env_name
# в_папке_проекта$ source env_name/bin/activate
# в_папке_проекта$ which python или pip
# должно показать папку виртуальной среды env_name
# в_папке_проекта$ pip install -r requirements.txt
# в_папке_проекта$ deactivate

# # ---------------------Поменяем фон ---------------------
set_background(os.path.join(IMG_PATH, 'fon.png'))


# ---------------------Текстовые элементы ---------------------
# st.header('Заголовок')
st.markdown('''<h1 style='text-align: center; color: black;'
            >Текстовые элементы:</h1>''', 
            unsafe_allow_html=True)

st.markdown('''<h2 style='text-align: center'>
                <span style='color: black;'>Разноцветный</span> <span style='color: #1FA7C9;'>заголовок</span> <span style='color: red'>2го</span> <span style='color: #119c2e;'>уровня</span>                
                </h2>
                ''', unsafe_allow_html=True)  

st.markdown(f''' \n##### Можно добавлять кликабельные ссылки в виде изображений: [<img src={'https://docs.streamlit.io/logo.svg'} width="45" >]({'https://docs.streamlit.io/library/api-reference'})
                        ''', unsafe_allow_html=True)

st.write("А можно делать то же самое в виде текстовой [ссылки](https://docs.streamlit.io/library/api-reference)")

with st.expander('"Экспандер" == раскрывающийся список:'):
    st.write('''
    \n Можно делать списки:
    \n 1. нумерованные
    \n * списки с маркерами 
    \n\t * вложенные списки через табуляцию "\\t" 
    \n\t\t * вложенные списки через табуляцию "\\t\\t" 

    \n А также другие элементы бибилиотеки streamlit можно вставлять в раскрывающийся список

    
    
    ''')
    st.image(os.path.join(IMG_PATH, 'ocr_example.png'))


st.write("""
Естественно, можно писать обычным текстом:

\n - _курсивом через подчеркивание_
\n - *курсивом через  "\*" с двух сторон*
\n - __жирным шрифтом через подчеркивание__
\n - **жирным шрифтом через "\**" с двух сторон**


А можно выделить текстовые элементы в красивые блоки:""")
         
st.caption('Например, эта сноска, поясняющая блок кода:')   


st.latex(r'''
    a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
    \sum_{k=0}^{n-1} ar^k =
    a \left(\frac{1-r^{n}}{1-r}\right)
    ''')
         


#-------------------------Визуальные элементы-------------------------
st.write("-------")
st.markdown('''<h1 style='text-align: center; color: black;'
            >Визуальные элементы:</h1>''', 
            unsafe_allow_html=True)
st.image(os.path.join(IMG_PATH, 'ocr_example.png'), use_column_width='auto', caption = 'Подпись к картинке')



# ---------------------Uploading img---------------------
st.write("-------")
st.markdown('''<h1 style='text-align: center; color: black;'
            >Распознавание текста 
            \n(c использованием библиотеки EasyOCR)
            </h1>''', 
            unsafe_allow_html=True) 
st.markdown(f''' \n##### Основная ссылка для работы с библиотекой: [<img src={'https://www.jaided.ai/static/img/svg_icon/EasyOCR_OSS3.svg'} width="70" >]({'https://www.jaided.ai/easyocr/tutorial/'})
                        ''', unsafe_allow_html=True)
st.markdown('''###### В этом блоке вы попробуете использовать библиотеку EasyOCR для распознавания текста с изображений.
\nДля этого:
\n(1) Выберете и загрузите изображение в формате jpg, jpeg или png. Это может быть скан документа, снимок страницы книги, фотография вывески и др.
\n(2) Выберете язык для распознавания. Библиотека EasyOCR работает более, чем с 80 языками. Для простоты, в нашей лабораторной работе мы ограничились 21 из них:''')
with st.expander('...список из 21 языка'):
    st.markdown('''
    \n'ar' - арабский,
    \n'az' - азербайджанский,
    \n'be' - беларусский,
    \n'bg' - болгарский,
    \n'ch_tra' - традиционный китайский,
    \n'che' - чеченский,
    \n'cs' - чешский,
    \n'de' - немецкий,
    \n'en' - английский,
    \n'es' - испанский,
    \n'fr' - французский,
    \n'hi' - хинди,
    \n'hu' - венгерский,
    \n'it' - итальянский,
    \n'ja' - японский, 
    \n'la' - латынь,
    \n'pl' - польский,
    \n'ru' - русский,
    \n'tr' - турецкий,
    \n'uk' - украинский,
    \n'vi' - вьетнамский
    ''')
st.markdown('''Полный же список распознаваемых EasyOCR языков смотрите [тут](https://www.jaided.ai/easyocr/#Supported%20Languages).
\n(3) После выбора языка начнётся его распознавание на изображении. Этот процесс займёт некоторое время.
\nВ начале распознавания вы увидите обработанное изображение с ограничивающими рамками (bounding boxes) вокруг тех участков, где нейронная сеть нашла буквы. 
Далее выведется сам распознанный текст.

\n_Обратите внимание:_
1. чем чётче загруженное изображение, тем точнее будет определён текст на нём
2. учитывайте также и особенности изображения: старославянскую рукопись библиотека не распознает, это задача для нейронных сетей более сложной архитектуры
3. при загрузке больших документов, время обработки существенно увеличится
4. неверный выбор языка для определения также может быть причиной ошибок''')
            
uploaded_img = st.file_uploader("Ниже загрузите изображение с текстом:", type=['jpg', 'jpeg', 'png'])
if uploaded_img is not None: 
    st.image(uploaded_img, use_column_width='auto', caption=f'Загруженное изображение {uploaded_img.name}')
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8) # переводим в numpy.ndarray
    bytearray_img = cv2.imdecode(file_bytes, 1) # переводим в numpy.ndarray

# ---------------------Choosing language---------------------
languages = ['ar','az','be','bg','ch_tra','che','cs','de','en','es','fr','hi','hu','it','ja','la','pl','ru','tr','uk','vi']
chose_lang = st.multiselect('Выберите язык для распознавания:', languages)

if st.button('Распознать текст с загруженного изображения'):
    if not chose_lang or not uploaded_img:
        st.write('_Обработка приостановлена: загрузите изображение и/или выберите язык для распознавания._')
    else:
        # экземпляр класса Reader - с чем будем работать в библиотеке EasyOCR:
        reader = Reader(chose_lang)

        # получаем координаты границ bounding-boxes:
        bounds = reader.readtext(bytearray_img) # работает c bytearray_img
        st.write(bounds)
        # рисуем границы bounding-boxes на изображении с помощью написанной ранее функции:
        boxes = draw_boxes(uploaded_img, bounds) # работает c uploaded_img
        st.image(boxes)

        # непосредственно распознаём текст с изображения:
        result = reader.readtext(bytearray_img, detail = 0, paragraph=True)
        
        st.markdown('##### Распознанный текст:')
        for string in result:
            st.write(string)

        result_as_str = ' '.join(result)

        st.download_button(label="Загрузить результат в формате '.txt'",
                        data=result_as_str,
                        file_name='Текст_из_streamlit_OCR.txt',
                        mime='text/csv')


#---------------------Дополнительные взаимодействия:---------------------
st.write("-------")
st.markdown('''<h1 style='text-align: center; color: black;'
            >Дополнительные взаимодействия
            </h1>''', 
            unsafe_allow_html=True) 
st.markdown('''##### Их можно создавать, как для проверки пользователя, так и для выбора пользоватлем различных условий при взаимодействии с приложением: ''')            
with st.form('Ответьте на все вопросы, чтобы успешно завершить лабораторную работу'):
    st.markdown('**Пример с элементом st.checkbox:**')
    question_1_right_1 = st.checkbox('Верный пункт', value=False, key='1')
    question_1_wrong_2 = st.checkbox('Неправильный ответ', value=False, key='2')
    question_1_right_3 = st.checkbox('Правильный ответ', value=False, key='3')

    st.markdown('**Пример с элементом st.radio:**')
    question_2 = st.radio('Выберите ответ:', ('не знаю', 'Неверный вариант', 'Единственный правильный вариант', 'Неправильный ответ'))

    right_answers = (question_1_right_1 and question_1_right_3 and question_2=='Единственный правильный вариант')
    wrong_answers = (question_1_wrong_2 or question_2=='Неверный вариант' or question_2=='Неправильный ответ')
    
    if st.form_submit_button('Закончить тест и посмотреть результаты'):
        if right_answers==True and wrong_answers==False:
            st.markdown('''<h3 style='text-align: left; color: green;'
            >Тест сдан! Лабораторная работа завершена.</h3>''', 
            unsafe_allow_html=True) 
        else:
            st.markdown('''<h3 style='text-align: left; color: red;'
            >Тест не сдан! Где-то была допущена ошибка.</h3>''', 
            unsafe_allow_html=True) 


    













        
        
        
       
