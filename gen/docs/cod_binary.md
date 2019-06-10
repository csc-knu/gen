{% include mathjax %}

## CodBinary
	
### Призначення

Процедура виконує кодування довільного дійсного числа $$x_{\text{dec}}$$ з заданого діапазону $$[x_{\text{min}}..x_{\text{max}}]$$ з заданою точністю $$\varepsilon$$ у послідовність з 0 і 1 фіксованої довжини. 

Процедура працює у парі з процедурою [CodDecimal](cod_decimal.md), яка виконує зворотнє перетворення. 

Допоміжні параметри обчислюються процедурою [BinDecParam](bin_dec_param.md).

### Вхідні параметри

- $$x_{\text{dec}}$$ &mdash; десяткове число;

- $$x_{\text{min}}$$ &mdash; мінімальне значення числа, що кодується;

- $$l$$ &mdash; ціле число, максимальна кількість двійкових розрядів для представлення довільного числа із заданого діапазону із заданою точністю;

- $$d$$ &mdash; дискретність кодування дійсного числа $$x_{\text{dec}}$$ цілим числом.

### Вихідні параметри

- $$X_{\text{bin}}$$ &mdash; список з $$l$$ розрядів двійкового числа, молодші розряді йдуть спочатку. 

У разі потреби старші розряди дозаповнюються нулями.

### Обчислення

Ціле число частин величини $$d$$ для заданого числа $$x_{\text{dec}}$$ можно обчислити як

$$
xx = \left[ \frac{x_{\text{dec}} - x_{\text{min}}}{d} \right].
$$

Ціле число $$xx$$ записуємо у двійковій формі і доповнюємо старші розряди нулями, якщо їхня кількість менше $$l$$.

### Вказівки

Значення $$l$$ і $$d$$ обчислюються процедурою [BinDecParam](bin_dec_param.md) і не можуть задаватися довільно.

Перетворення цілого десяткового числа у двійковий код (список 0 і 1) можна виконати функцією convert(xx, base, 2).

[Назад до лаби](README.md)

[Назад на головну](../README.md)