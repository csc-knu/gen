{% include mathjax %}

## Бджолиний алгоритм

### Неформальний опис алгоритму

Нехай, маємо стандартну задачу оптимізації (максимізації) функції $$f(x)$$ в просторі розв'язків $$\RR^n$$.

Алгоритм бджолиного пошуку використовує в своїй роботі наступні числові парамери:

- $$s$$ &mdash; число бджіл-розвідників;

- $$p$$ &mdash; число вибраних точок (розв'язків) для подальшого дослідження ($$p < s$$);

- $$e$$ &mdash; число найкращих (элітних) точок ($$e < p$$);

- $$s_e$$ &mdash; число бджол для більш повного дослідження $$e$$ елітних розв'язків;

- $$s_p$$ &mdash; число бджол для більш повного дослідження решти $$(p − e)$$ вибраних розв'язків;

- $$\delta$$ &mdash; розмір околу, у якому бджоли здійснюють детальніший пошук.

Таким чином, алгоритм складається з наступних кроків. 

Спочатку випадковим чином обираємо $$m$$ розв'язків, кожний з яких є одною бджолою-розвідником. 

Найкращі $$m$$ розв'язки досліджуються більш детально (локальний пошук). 

Для цього в їх $$\delta$$-околах розглядаються $$r$$ випадкових точок ($$r = s_e$$ або $$r = s_p$$ в залежності від _рівня_ елітності данної точки), з яких обирається одна найкраща. 

Решту $$(s − m)$$ розв'язків заміняються на випадкові з простору розв'язків (випадковий пошук). 

Алгоритм завершуємо, коли виявляється виконаним якась умова зупинки (досягнута необхідна точність, пройдена достатня кількість ітерацій, і т.&nbsp;д.)

### Код

- Параметри алгоритму:

```maple
restart;
s := 10;
m := 8;
e := 6;
s_e := 6; 
s_p := 4; 
lambda := 1.33;
Bind := 5.12;
iterations := 10;
```

- Параметри задачі:

```maple
N := 3;
F := proc(x, y, z) -> 10 * N + x^2 - 10 * cos(2 * Pi * x) + y^2 - 10 * cos(2 * Pi * y) + z^2 - 10 * cos(2 * Pi * z); 
Rand := proc(x, y) -> RandomTools[Generate](float(range = x..y)); 

X := Matrix(s, N, proc(i, j) -> Rand(-Bind, Bind));
x := proc(i) -> seq(X[i, j], j = 1..N);
```

- Основний алгоритм

```maple
itr := 0;
while itr < iterations do
    itr := itr + 1;
    for i from 1 to (s - 1) do
        for j from 1 to (s - i) do
            if evalf(F(x(j))) > evalf(F(x(j + 1))) then
                for k from 1 to N do
                    tmp := x(j);
                    X[j, k] := X[j + 1, k];
                    X[j + 1, k] := tmp[k];
                end do;
            end if;
        end do;
    end do;

    r := 0;

    for i from 1 to m do
        if i <= e then
            r := s_e;
        end if;

        if i > e then
            r := s_p;
        end if;

        xBest := x(i);

        for j from 1 to r do
            print(j)
            print(lambda);
            left := seq(X[i, k] - lambda, k = 1..N);
            print(left);
            right := seq(X[i, k] + lambda, k = 1..N);
            print(right);

            xRand1 := seq(RandomTools[Generate](float(range = -1..1)), k = 1..N);
            xRand := seq((right[k] - left[k]) / 2 * xRand1[k]) + (right[k] + left[k]) / 2, k = 1..N);

            print(xRand);
            print(j);

            if evalf(F(xBest)) > evalf(F(xRand)) then
                xBest := xRand;
            end if;
        end do;

        for k from 1 to N do
            X[i, k] := xBest[k];
        end do;
    end do;

    for i from m + 1 to s do
        for j from 1 to N do
            X[i, j] := Rand(-Bind, Bind);
        end do;
    end do;

    print(x(1));

end do;
```
#### Графіки

Результати при

- 10 ітераціях:

	

- 100 ітераціях:

	

- 1000 ітераціях:

	


[Назад на головну](../README.md)
