from flask import Flask, render_template, request, redirect, url_for, session
import re
import requests
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
import random
import plotly.io as pio
import datetime 
import openai
import os
import plotly.graph_objs as go
import pymysql.cursors
from werkzeug.security import check_password_hash
import pymysql
import hashlib  # hashlib kütüphanesini ekleyin
import time
import threading  # threading modülünü ekleyin
import transformers


app = Flask(__name__, template_folder="templates")
app.secret_key = 'jjkmloading741852'  # Güvenlik için gizli anahtarınızı ayarlayın

# MySQL bağlantı bilgilerini yapılandırın
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = '3306'
app.config['MYSQL_USER'] = ''
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = ''

# MySQL bağlantısını oluşturun
mysql = pymysql.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB'],
    cursorclass=pymysql.cursors.DictCursor
)

last_refresh_time = time.time()
refresh_interval = 60  # 60 saniyede bir yenile

# MySQL bağlantısını yenilemek için bir işlev tanımlayın
def refresh_mysql_connection():
    global last_refresh_time

    while True:
        if (time.time() - last_refresh_time) > refresh_interval:
            try:
                mysql.ping(reconnect=True)
                last_refresh_time = time.time()
            except Exception as e:
                print(f"MySQL bağlantısını yenileme hatası: {str(e)}")

        time.sleep(1)  # Her saniye kontrol et

# Yenileme işlemini başlatın
refresh_thread = threading.Thread(target=refresh_mysql_connection)
refresh_thread.daemon = True
refresh_thread.start()


def fetch_reviews(product_id, page):
    base_url = "https://public-mdc.trendyol.com/discovery-web-socialgw-service/api/review/"
    api_url = f"{base_url}{product_id}?page={page}"
    response = requests.get(api_url)
    data = response.json()
    return data

def get_all_reviews(product_id):
    page = 0
    all_reviews = []

    while True:
        data = fetch_reviews(product_id, page)
        if (
            "result" in data
            and "productReviews" in data["result"]
            and data["result"]["productReviews"] is not None
            and data["result"]["productReviews"]["content"]
        ):
            reviews = data.get("result", {}).get("productReviews", {}).get("content", [])
            all_reviews.extend(reviews)
            page += 1
        else:
            break

    return all_reviews
    
    
api_key = ""
openai.api_key = api_key

# generate_product_analysis işlevini Flask uygulamanızın dışında tanımlayın
def generate_product_analysis(text):
    # GPT-3.5-Turbo ile tüm yorumları analiz edin
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Özetle: " + text,
        temperature=0,
        max_tokens=500  # İstenilen maksimum metin uzunluğunu ayarlayabilirsiniz
    )

    analysis_result = response.choices[0].text

    return analysis_result

@app.route('/productDetail/<product_id>', methods=['GET', 'POST'])
def product_detail(product_id):
    all_reviews = get_all_reviews(product_id)

    df_data = []
    for review in all_reviews:
        df_data.append({
            "commentDateISOtype": review.get("commentDateISOtype", ""),
            "rate": review.get("rate", ""),
            "comment": review.get("comment", ""),
            "sellerName": review.get("sellerName", "")
        })

    df = pd.DataFrame(df_data)

    if request.method == 'POST':
        selected_sellers = request.form.getlist('sellerNames')
        selected_rates = request.form.getlist('rate')
        sort_option = request.form.get('sort')  # Hangi sıralama türünün seçildiğini alın

        filtered_df = df
        if selected_sellers:
            filtered_df = filtered_df[filtered_df['sellerName'].isin(selected_sellers)]
        if selected_rates:
            selected_rates = [int(rate) for rate in selected_rates]
            filtered_df = filtered_df[filtered_df['rate'].isin(selected_rates)]

        # DataFrame'in son 20 satırını seçin
        last_20_reviews_df = filtered_df.iloc[-70:]

        # Tüm yorumları birleştirin ve başlık cümlesini ekleyin
        all_reviews_text = "***"
        all_reviews_text += " ".join(last_200_reviews_df["comment"].tolist())

        # Ürün analizini oluşturun
        analysis_result = generate_product_analysis(all_reviews_text)

        seller_average_ratings = filtered_df.groupby("sellerName").agg(
            TotalReviews=pd.NamedAgg(column="rate", aggfunc="count"),
            AverageRating=pd.NamedAgg(column="rate", aggfunc="mean")
        ).reset_index()

        # Seller verilerini AverageRating'e göre sıralayın
        sorted_seller_df = seller_average_ratings.sort_values(by='AverageRating', ascending=False)

        # Index'i sıfırdan başlayacak şekilde yeniden düzenleyin
        sorted_seller_df.reset_index(drop=True, inplace=True)

        comments = all_reviews_text  # Tüm yorumları kullanarak analiz yaptık
        words = nltk.word_tokenize(comments)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stopwords.words("turkish")]
        words = [word for word in words if word not in ["çok", "bir", "bi", "ama", "yani", "hiç", "ürün", "urun", "ve", "ederim", "teşekkür", "bile", "ben", "olarak", "sadece"]]

        word_counts = Counter(words)
        most_common_words = word_counts.most_common(10)

        # Tüm yorumlardaki puan sayısını hesapla
        rating_counts = filtered_df['rate'].value_counts().to_dict()

        # Plotly ile pasta grafik oluştur
        fig = go.Figure(data=[
            go.Pie(
                labels=list(rating_counts.keys()),
                values=list(rating_counts.values()),
                hoverinfo='label+percent+value+name',  # Adet, yüzde, değer ve etiketi gösterir
            )
        ])

        fig.update_layout(title='Ürünün Puan Dağılımları',
                          title_x=0.5  # Başlığı yatayda ortala (0.5)
                          )

        # Pasta grafiği HTML'e dönüştürerek gönderin
        pie_chart_html = fig.to_html(full_html=False)

        # Yorumlara göre aylık yorum sayısını çizin
        filtered_df['commentDateISOtype'] = pd.to_datetime(filtered_df['commentDateISOtype'])
        filtered_df['commentMonth'] = filtered_df['commentDateISOtype'].dt.month
        monthly_comment_counts = filtered_df['commentMonth'].value_counts().sort_index()

        # Çizgi grafiği oluşturun
        line_fig = go.Figure(data=go.Scatter(x=monthly_comment_counts.index, y=monthly_comment_counts.values, mode='lines+markers'))

        # Grafiği özelleştirin
        line_fig.update_layout(
            title='Aylara Göre Yorum Sayısı',
            title_x=0.5,
            xaxis=dict(title='Ay'),
            yaxis=dict(title='Yorum Sayısı'),
            xaxis_tickvals=list(range(1, 13)),  # 1'den 12'ye kadar olan aylar için tick değerleri
            xaxis_ticktext=['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']  # Ay isimleri
        )

        # Çizgi grafiği HTML'e dönüştürerek gönderin
        line_chart_html = line_fig.to_html(full_html=False)
        
        # GET isteği durumunda
        seller_average_ratings = df.groupby("sellerName").agg(
            TotalReviews=pd.NamedAgg(column="rate", aggfunc="count"),
            AverageRating=pd.NamedAgg(column="rate", aggfunc="mean")
        ).reset_index()

        # En yüksek inceleme alan satıcıyı bulun
        highest_rated_seller = seller_average_ratings.loc[seller_average_ratings['TotalReviews'].idxmax()]

        
            # Satıcıların toplam inceleme sayılarını hesaplayın
        seller_total_reviews = seller_average_ratings.groupby("sellerName")["TotalReviews"].sum().reset_index()

        # Pasta grafik için verileri hazırlayın
        fig = px.pie(seller_total_reviews, values='TotalReviews', names='sellerName', title='Satıcıların Satış Adetleri Dağılımı')

        # Pasta grafiğini HTML'e dönüştürerek gönderin
        pie_chart_htmle = fig.to_html(full_html=False)


        return render_template('productDetail.html', df_table=df.to_html(), sorted_sellers_table=sorted_seller_df.to_html(), most_common_words=most_common_words, df=df, rating_counts=rating_counts, analysis_result=analysis_result, pie_chart=pie_chart_html, line_chart=line_chart_html, highest_rated_seller=highest_rated_seller, pie_charte=pie_chart_htmle)

    # GET isteği durumunda
    seller_average_ratings = df.groupby("sellerName").agg(
        TotalReviews=pd.NamedAgg(column="rate", aggfunc="count"),
        AverageRating=pd.NamedAgg(column="rate", aggfunc="mean")
    ).reset_index()

    comments = " ".join(df["comment"].tolist())  # Tüm yorumları birleştirdik
    words = nltk.word_tokenize(comments)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords.words("turkish")]
    words = [word for word in words if word not in ["çok", "bir", "bi", "ama", "yani", "hiç", "ürün", "urun", "ve", "ederim", "teşekkür"]]

    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)

    # Tüm yorumlardaki puan sayısını hesapla
    rating_counts = df['rate'].value_counts().to_dict()

    # Plotly ile pasta grafik oluştur
    fig = go.Figure(data=[
        go.Pie(
            labels=list(rating_counts.keys()),
            values=list(rating_counts.values()),
            hoverinfo='label+percent+value+name',  # Adet, yüzde, değer ve etiketi gösterir
        )
    ])

    fig.update_layout(title='Ürünün Puan Dağılımları',
                      title_x=0.5  # Başlığı yatayda ortala (0.5)
                      )

    # Pasta grafiği HTML'e dönüştürerek gönderin
    pie_chart_html = fig.to_html(full_html=False)

    # Yorumlara göre aylık yorum sayısını çizin
    df['commentDateISOtype'] = pd.to_datetime(df['commentDateISOtype'])
    df['commentMonth'] = df['commentDateISOtype'].dt.month
    monthly_comment_counts = df['commentMonth'].value_counts().sort_index()

    # Çizgi grafiği oluşturun
    line_fig = go.Figure(data=go.Scatter(x=monthly_comment_counts.index, y=monthly_comment_counts.values, mode='lines+markers'))

    # Grafiği özelleştirin
    line_fig.update_layout(
        title='Aylara Göre Yorum Sayısı',
        title_x=0.5,
        xaxis=dict(title='Ay'),
        yaxis=dict(title='Yorum Sayısı'),
        xaxis_tickvals=list(range(1, 13)),  # 1'den 12'ye kadar olan aylar için tick değerleri
        xaxis_ticktext=['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']  # Ay isimleri
    )

    # Çizgi grafiği HTML'e dönüştürerek gönderin
    line_chart_html = line_fig.to_html(full_html=False)
    
        # GET isteği durumunda
    seller_average_ratings = df.groupby("sellerName").agg(
        TotalReviews=pd.NamedAgg(column="rate", aggfunc="count"),
        AverageRating=pd.NamedAgg(column="rate", aggfunc="mean")
    ).reset_index()

    # En yüksek inceleme alan satıcıyı bulun
    highest_rated_seller = seller_average_ratings.loc[seller_average_ratings['TotalReviews'].idxmax()]


    
    # Satıcıların toplam inceleme sayılarını hesaplayın
    seller_total_reviews = seller_average_ratings.groupby("sellerName")["TotalReviews"].sum().reset_index()

        # Pasta grafik için verileri hazırlayın
    fig = px.pie(seller_total_reviews, values='TotalReviews', names='sellerName', title='Satıcıların Satış Adetleri Dağılımı')

        # Pasta grafiğini HTML'e dönüştürerek gönderin
    pie_chart_htmle = fig.to_html(full_html=False)

    return render_template('productDetail.html', df_table=df.to_html(), sorted_sellers_table=seller_average_ratings.to_html(), most_common_words=most_common_words, df=df, rating_counts=rating_counts, pie_chart=pie_chart_html, line_chart=line_chart_html, highest_rated_seller=highest_rated_seller, pie_charte=pie_chart_htmle)

@app.route('/save_url', methods=['POST'])
def save_url():
    if session.get('logged_in'):
        entered_url = request.form.get('url')

        # URL'yi veritabanında kontrol edin, eğer kayıtlı değilse kaydedin
        with mysql.cursor() as cursor:
            cursor.execute("SELECT user_email, magaza_url FROM wp_users WHERE user_email = %s", (session['user_email'],))
            user = cursor.fetchone()

            if user:
                magaza_url = user['magaza_url']

                if not magaza_url:
                    cursor.execute("UPDATE wp_users SET magaza_url = %s WHERE user_email = %s", (entered_url, session['user_email']))
                    mysql.commit()
    
    return redirect(url_for('index'))


app.permanent_session_lifetime = datetime.timedelta(seconds=3600)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None  # Hata mesajını saklamak için bir değişken

    if request.method == 'POST':
        username = request.form['username']
        entered_password = request.form['password']

        try:
            # MySQL sorgusuyla kullanıcıyı kontrol edin
            with mysql.cursor() as cursor:
                cursor.execute("SELECT user_email, user_p FROM wp_users WHERE user_email = %s", (username,))
                user = cursor.fetchone()

                if user:
                    stored_password = user['user_p']

                    # Kullanıcının girdiği şifreyi doğrudan karşılaştırın
                    if entered_password == stored_password:
                        # Kullanıcı girişi başarılı, oturumu başlatın
                        session.permanent = True  # Oturumu kalıcı yapın
                        session['logged_in'] = True
                        session['user_email'] = username  # Kullanıcının adını oturum verisine ekleyin
                        return redirect(url_for('index'))  # Başarılı giriş durumunda 'index' sayfasına yönlendir
                    else:
                        # Kullanıcı girişi başarısız, hata mesajını ayarlayın
                        error = 'Yanlış Şifre'
                else:
                    # Kullanıcı adı bulunamadı, hata mesajını ayarlayın
                    error = 'Kullanıcı adı bulunamadı.'
        except Exception as e:
            # Veritabanı hatası oluştuğunda hata mesajı gösterin
            error = 'Veritabanı hatası: {}'.format(str(e))
    
    return render_template('login.html', error=error)


# Çıkış işlevi
@app.route('/logout')
def logout():
    # Oturumu sonlandırın
    session.pop('user_email', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    # Oturum kontrolü yaparak sadece oturumu açık olan kullanıcılara erişim izni verin
    if not session.get('user_email'):
        return redirect(url_for('login'))  # Oturumu açık olmayan kullanıcıları /login sayfasına yönlendirin
    magaza_url = None  # Default olarak magaza_url'i boş kabul ediyoruz
    user_login = None

# Burada veritabanından magaza_url bilgisini çekerek magaza_url değişkenine atayın
    with mysql.cursor() as cursor:
        try:
            cursor.execute("SELECT user_email, magaza_url, user_login FROM wp_users WHERE user_email = %s", (session['user_email'],))
        except pymysql.OperationalError as e:
            if e.args[0] == 2006:
                # MySQL sunucu bağlantısı kopmuş, yeniden kurulum yapılabilir.
                mysql.ping(reconnect=True)
        user = cursor.fetchone()
        if user:
            magaza_url = user['magaza_url']
            user_login = user['user_login']
    

    if request.method == 'POST':
        url = request.form.get('url')

        match = re.search(r"m-(\d+)", url)
        if match:
            sellerNo = match.group(1)

            filtered_data = []
            average_ratings = []
            page_index = 0

            while True:
                new_url = "https://searchpublic.trendyol.com/discovery-sellerstore-websfxsearch-santral/v1/search/products?q=sr/?mid=" + str(sellerNo) + "&pageIndex=" + str(page_index)

                response = requests.get(new_url)
                if response.status_code == 200:
                    api_data = response.json()

                    if not api_data:
                        break

                    for item in api_data:
                        if 'rating' in item and 'total' in item['rating'] and 'average' in item['rating']:
                            total_rating = item['rating']['total']
                            average_rating = '{:.2f}'.format(item['rating']['average'])
                            if total_rating > 1:
                                filtered_data.append({
                                    'id': item['id'],
                                    'brand': item['brand'],
                                    'title': item['title'],
                                    'current': item['price']['current'],
                                    'total_rating': total_rating,
                                    'average_rating': average_rating,
                                    'link': "https://www.trendyol.com" + item['link']
                                })
                                average_ratings.append(float(average_rating))
                    page_index += 1
                else:
                    return "API isteği başarısız oldu."

            sellerNo = match.group(1)
            page = 0
            page_size = 100
            
            url = f"https://public-sdc.trendyol.com/discovery-sellerstore-webgw-service/v1/ugc/seller-reviews/reviews/{sellerNo}?page={page}&pageSize={page_size}"

            # Tüm sayfaları dolaşmak için bir döngü
            all_data = []
            while True:
                url = f"https://public-sdc.trendyol.com/discovery-sellerstore-webgw-service/v1/ugc/seller-reviews/reviews/{sellerNo}?page={page}&pageSize={page_size}"
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    seller_reviews = data.get("sellerReviews", {}).get("content", [])
                    all_data.extend(seller_reviews)

                    # Bir sonraki sayfa varsa ilerle, yoksa döngüyü sonlandır
                    if data.get("sellerReviews", {}).get("totalPages", 0) > page + 1:
                        page += 1
                    else:
                        break
                else:
                    return jsonify({"error": "API isteği başarısız oldu. Hata kodu: " + str(response.status_code)})

            # Tüm verileri bir DataFrame'e dönüştürme
            df = pd.DataFrame(all_data)

            # Gereksiz sütunları kaldırma (Opsiyonel)
            columns_to_drop = ["forbiddenStorage", "lastModifiedDate", "lastEditedDate", "platform", "reviewStatus", "userFullName", "shipmentNumber", "showUserName", "orderType", "deliveryDate", "contentImages", "customerSatisfactionResponse", "votes", "mediaFiles"]
            df.drop(columns=columns_to_drop, inplace=True)

            # Ratings sütununu düzenleme
            df['SELLER_Rate'] = df['ratings'].apply(lambda x: next((item['vote']['rate'] for item in x if item['voteType'] == 'SELLER'), None))
            df['CARGO_Rate'] = df['ratings'].apply(lambda x: next((item['vote']['rate'] for item in x if item['voteType'] == 'CARGO'), None))

            # 'ratings' sütununu kaldırma
            df.drop(columns=['ratings'], inplace=True)

            # Unix zaman damgasını tarihe dönüştürme işlevi
            def convert_unix_timestamp_to_date(timestamp):
                return datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%d/%m/%Y')

            # CreationDate sütununu tarihe dönüştürme
            df['creationDate'] = df['creationDate'].apply(convert_unix_timestamp_to_date)

            # Ay isimleri
            turkish_month_names = [
                'Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık'
            ]

            # Gün isimleri
            turkish_day_names = [
                'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'
            ]

            # Ay ve gün isimlerini içeren yeni sütunları oluşturma
            df['Month'] = df['creationDate'].apply(lambda x: turkish_month_names[int(x.split('/')[1]) - 1])
            df['Day_Name'] = df['creationDate'].apply(lambda x: turkish_day_names[pd.Timestamp(x).weekday()])

            # Ay Dağılımı Grafiği
            month_order = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']  # Ayları sıralıyoruz
            month_counts = df['Month'].value_counts().reset_index()
            month_counts.columns = ['Month', 'Count']
            month_counts['Month'] = pd.Categorical(month_counts['Month'], categories=month_order, ordered=True)
            month_counts = month_counts.sort_values(by='Month')
            month_fig = px.line(month_counts, x='Month', y='Count', markers=True, labels={'Month': 'Ay', 'Count': 'Yorum Sayısı'})
            month_fig.update_layout(title_text='Aylara Göre Yorum Adetleri', title_x=0.5)
            month_plot = pio.to_html(month_fig, full_html=False)

            # Gün Adı Dağılımı Grafiği
            day_order = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']  # Günleri sıralıyoruz
            day_name_counts = df['Day_Name'].value_counts().reset_index()
            day_name_counts.columns = ['Day_Name', 'Count']
            day_name_counts['Day_Name'] = pd.Categorical(day_name_counts['Day_Name'], categories=day_order, ordered=True)
            day_name_counts = day_name_counts.sort_values(by='Day_Name')
            day_name_fig = px.bar(day_name_counts, x='Day_Name', y='Count', labels={'Day_Name': 'Günler', 'Count': 'Yorum Sayısı'})
            day_name_fig.update_layout(title_text='Günlere Göre Yorum Adetleri', title_x=0.5)
            day_name_plot = pio.to_html(day_name_fig, full_html=False)

            if filtered_data:
                df = pd.DataFrame(filtered_data)
                sorted_df = df.sort_values(by='average_rating').head(3000)
                # Ortalama değeri hesapla
                average_rating_mean = sum(average_ratings) / len(average_ratings)

                # En yüksek ve en düşük puanlı ürünleri bul
                highest_rated_product = sorted_df.iloc[-1]
                lowest_rated_product = sorted_df.iloc[0]

                # Puanların yüzde cinsinden dağılımını gösteren çubuk grafik oluştur
                plot_data = create_rating_distribution_plot(average_ratings)  # ratings listesini kullanarak

                bubble_chart = create_average_rating_bubble_chart(sorted_df)

                return render_template('deneme.html', df=df, table=sorted_df.to_html(index=False, escape=False, render_links=True),
                       average_rating_mean=average_rating_mean, highest_rated_product=highest_rated_product,
                       lowest_rated_product=lowest_rated_product, plot_data=plot_data, bubble_chart=bubble_chart, month_plot=month_plot, day_name_plot=day_name_plot, sorted_df=sorted_df)
            else:
                return "Veri bulunamadı."
        else:
            return "Mağaza Bilgisi Yok."

    return render_template('index.html', magaza_url=magaza_url, user_login=user_login)


def create_rating_distribution_plot(average_ratings):
    ratings_count = [0, 0, 0, 0, 0]  # Her grup için başlangıçta 0 sayısı
    total_ratings = len(average_ratings)  # Toplam puan sayısı

    for rating in average_ratings:
        if 1 <= rating < 1.5:
            ratings_count[0] += 1
        elif 1.5 <= rating < 2.5:
            ratings_count[1] += 1
        elif 2.5 <= rating < 3.5:
            ratings_count[2] += 1
        elif 3.5 <= rating < 4.5:
            ratings_count[3] += 1
        elif 4.5 <= rating <= 5:
            ratings_count[4] += 1

    labels = ['1 Yıldız', '2 Yıldız', '3 Yıldız', '4 Yıldız', '5 Yıldız']
    ratings = [1, 2, 3, 4, 5]
    

    fig = px.bar(x=ratings_count, y=ratings, orientation='h', text=ratings_count, labels={'x':'Adet', 'y':'Puan'})

    # Yüzde oranlarını ve adetleri ekle
    for i, count in enumerate(ratings_count):
        fig.add_annotation(text=f'({(count / total_ratings * 100):.1f}%)', x=count + 10, y=i + 1, showarrow=False)

    # Başlığı ortala
    fig.update_layout(
        title_text='Mağaza Puan Dağılımı',
        title_x=0.5  # 0.5, başlığı yatayda ortalar
    )

    plot_data = pio.to_html(fig, full_html=False)
    
    return plot_data
    
def create_average_rating_bubble_chart(data):
    data = data.rename(columns={
        'Marka': 'brand',
        'Başlık': 'title',
        'Fiyat': 'current',
        'Toplam Değerlendirme': 'total_rating',
        'Ürün Puanı': 'average_rating'
    })

    fig = px.scatter(data, x='average_rating', y='total_rating', color='average_rating',
                     size='total_rating', hover_data=['brand', 'title', 'current'])

    fig.update_layout(
        title='Ürün Değerlendirme Detayları',
        xaxis_title='Ürün Puanı',
        yaxis_title='Toplam Değerlendirme',
        showlegend=False,
        title_x=0.5  # Başlığı yatayda ortala (0.5)
    )

    return pio.to_html(fig, full_html=False)

if (time.time() - last_refresh_time) > 120:
    mysql.ping(reconnect=True)
    last_refresh_time = time.time()

if __name__ == '__main__':
    app.run(debug=True)
