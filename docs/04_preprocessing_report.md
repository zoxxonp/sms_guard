# Задание 4. Отчёт о предварительной обработке SMS

*Сгенерирован скриптом `python -m ml.data.clean_sms`, 07.11.2026.*

Функция обработки — `ml/preprocessing.py::normalize_sms`. Она же используется при обучении моделей и в веб-приложении.

## 1. Шаги обработки

Шаги 2–12 не удаляют записи, а изменяют текст; в столбце указано, у скольких сообщений шаг что-то изменил (одна запись может попасть в несколько шагов).

| № | Шаг | Затронуто записей | Примечание |
| :-: | :--- | --: | :--- |
| 1 | Исходные записи (sms_raw.csv) | 5574 |  |
| 2 | Unicode-нормализация NFKC изменила текст | 15 | полноширинные символы, лигатуры, неразрывные пробелы |
| 3 | Раскодированы HTML-сущности (`&lt;` `&gt;` `&amp;`) | 309 | метки `&lt;#&gt;`, `&lt;DECIMAL&gt;`, `&lt;TIME&gt;` → `<num>`, `&lt;URL&gt;` → `<url>` |
| 4 | Исправлены / удалены управляющие и невидимые символы | 35 | C1-символы cp1252 (\x92 и др.) → апострофы и кавычки; прочие Cc, Cf → пробел |
| 5 | Приведение к нижнему регистру (были заглавные буквы) | 5486 |  |
| 6 | URL → `<url>` | 155 |  |
| 7 | E-mail → `<email>` | 8 |  |
| 8 | Сумма с валютой → `<money>` | 425 | £, $, €, ₸, тг, руб, pence |
| 9 | Телефон → `<phone>` | 426 | 7–15 цифр |
| 10 | Прочие числа → `<num>` | 1336 |  |
| 11 | «!» / «?» выделены как признаки давления | 1917 | сохраняются отдельными токенами |
| 12 | Схлопнуты повторяющиеся пробелы | 260 |  |
| 13 | Удалены строки с пустым текстом | 0 | после очистки |
| 14 | Удалены дубликаты по text_clean | 439 | ham — 311, spam — 128 |
| 15 | Итог (sms_clean.csv) | 5135 |  |

## 2. Классы до и после очистки

| Класс | До очистки | После очистки | Удалено | Доля после, % |
| :--- | --: | --: | --: | --: |
| ham (0) | 4827.0 | 4516.0 | 311.0 | 87.95 |
| spam (1) | 747.0 | 619.0 | 128.0 | 12.05 |

## 3. Дубликаты по классам

| Класс | Точных дубликатов исходного текста | Удалено дубликатов по `text_clean` |
| :--- | --: | --: |
| ham (0) | 309 | 311 |
| spam (1) | 94 | 128 |
| **Всего** | **403** | **439** |

Дубликатов по `text_clean` больше, чем по исходному тексту: после замены чисел, телефонов и сумм токенами сообщения-шаблоны, отличающиеся только номером или суммой, становятся одинаковыми. Для модели это один и тот же пример, поэтому он оставлен один раз (первое вхождение).

Текстов, встречающихся с разными метками (ham и spam одновременно): **0**. 

Дубликаты удалены **до** разбиения на выборки (задание 6), чтобы одно и то же сообщение не попало одновременно в обучающую и тестовую выборки.

## 4. Примеры преобразования

| id | Класс | text_raw | text_clean |
| --: | :-: | :--- | :--- |
| 1 | 0 | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat... | `go until jurong point, crazy.. available only in bugis n great world la e buffet... cine there got amore wat...` |
| 3 | 1 | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's | `free entry in <num> a wkly comp to win fa cup final tkts 21st may <num> . text fa to <num> to receive entry question(std txt rate)t&c's apply <phone> over18's` |
| 7 | 0 | Even my brother is not like to speak with me. They treat me like aids patent. | `even my brother is not like to speak with me. they treat me like aids patent.` |
| 9 | 1 | WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only. | `winner ! ! as a valued network customer you have been selected to receivea <money> prize reward ! to claim call <phone> . claim code kl341. valid <num> hours only.` |
| 12 | 1 | SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info | `six chances to win cash ! from <num> to <money> txt> csh11 and send to <num> . cost <money> /day, 6days, <num> + tsandcs apply reply hl <num> info` |
| 16 | 1 | XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL | `xxxmobilemovieclub: to use your credit, click the wap link in the next txt message or click here>> <url> <url>` |
| 13 | 1 | URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18 | `urgent ! you have won a <num> week free membership in our <money> prize jackpot ! txt the word: claim to no: <num> t&c <url> lccltd pobox 4403ldnw1a7rw18` |

## 5. Стоп-слова

Удаление стоп-слов реализовано как опция: `normalize_sms(text, remove_stop_words=True)` или `remove_stopwords(text)`. В `sms_clean.csv` стоп-слова **не удалены**: их влияние на качество сравнивается в задании 8. Списки: английский — scikit-learn, русский и казахский — служебные слова, заданные в модуле (без загрузки корпусов NLTK).
