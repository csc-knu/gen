{% include mathjax %}

## Опис алгоритму

Нехай є певна фітнес-функція $$f: \mathbb{R}^m \to \mathbb{R}$$ і ставиться оптимцізаційна задача

$$
f(x) \xrightarrow[x \in \mathcal{C}]{} \min,
$$

де $$\mathcal{C} \subset \mathbb{R}^m$$ &mdash; опукла і замкнена допустима множина, наприклад $$\mathcal{C} = [x_{\text{min}}, x_{\text{max}}]^m$$.

Розглянемо популяцію з $$n$$ особин які протягом $$M$$ поколінь розв'язують цю оптимізаційну задачу (пристосовуються під задану фітнес-функцію). Кожну особину будемо описувати двійковим вектором достатньо довгим для того щоб кодувати десяткові значення аргументів функції $$x_i$$ з потрібною точністю $$\varepsilon$$.

Перше покоління не має &laquo;досвіду предків&raquo; і генерується випадковим чином, рівномірно на $$\mathcal{C}$$.

Далі з кожним поколінням відбуваються наступні речі:

- воно кодується з десяткового представлення у двійкове;

- у ньому із заданою ймовірністю $$p$$ відбувають бітові мутації;

- покоління розбивається на пари, між якими відбувається кроссовер (обмін генами);

- на основі покоління генерується нове таким чином що найкращі особини мають більшу ймовірність мати нащадка.

## Псевдокод

_Точніше не псевдокод а програма-драйвер на python._

```python
g_dec = generation_dec(n, x_min, x_max)
nn, dd, NN = bin_dec_param(x_min, x_max, eps)

for _ in range(M):
	g_bin = a_cod_binary(g_dec, x_min, nn, dd)
	g_bin, mutation_count = mutation(g_bin, p)
	m, f = parents(n >> 1)
	g_bin = crossover(g_bin, m, f)
	g_dec = a_cod_decimal(g_bin, x_min, NN, dd)
	f_vals = np.array([f(g_dec[i]) for i in range(n)]).reshape((n, 1))
	g_dec = np.hstack([g_dec, f_vals])
	b, w = best(g_dec), worst(g_dec)
	g_best = np.asarray(g_dec[b, :-1]).flatten()
	g_dec = np.delete(g_dec, w, axis=0)
	num = adapt(np.asarray(g_dec[:, -1]).flatten())
	g_dec = np.vstack([new_generation(g_dec[:, :-1], num), g_best])
``` 

## Реалізація

_Для зручності реалізації і оцінювання специфікація всіх необхідних процедур вже визначена, але від неї можна несуттєво відходити якщо так буде зручніше програмувати на вибраній Вами мові програмування._

_Також присутні вказівки для Maple і приклади реалізації на python._

Резюмуючи, Вам необхідно написати наступні процедури:

1. GenerationDec: [документація](docs/generation_dec.md), [приклад реалізації](code/generation_dec.py);

2. Mutation: [документація](docs/mutation.md), [приклад реалізації](code/mutation.py);

3. Crossover: [документація](docs/crossover.md), [приклад реалізації](code/crossover.py);

4. Parents: [документація](docs/parents.md), [приклад реалізації](code/parents.py);

5. BinDecParam: [документація](docs/bin_dec_param.md), [приклад реалізації](code/bin_dec_param.py);

6. CodBinary: [документація](docs/cod_binary.md), [приклад реалізації](code/cod_binary.py);

7. CodDecimal: [документація](docs/cod_decimal.md), [приклад реалізації](code/cod_decimal.py);

8. ACodBinary: [документація](docs/a_cod_binary.md), [приклад реалізації](code/cod_binary.py);

9. ACodDecimal: [документація](docs/a_cod_decimal.md), [приклад реалізації](code/cod_decimal.py);

10. Adapt: [документація](docs/adapt.md), [приклад реалізації](code/adapt.py);

11. Best і Worst: [документація](docs/best_worst.md), [приклад реалізації](code/best_worst.py);

12. NewGeneration: [документація](docs/new_generation.md), [приклад реалізації](code/new_generation.py);

І, нарешті, написати програму-драйвер, яка буде поєднувати усі вищезгадані процедури.

[Назад на головну](../README.md)