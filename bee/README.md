{% include mathjax %}

## Бджолиний алгоритм

### Неформальний опис алгоритму

### Код

```maple
itr := 1;
while itr < 10 do
    itr := itr + 1;
    for i from 1 to (s - 1) do
        for j from 1 to (s - i) do
            if evalf(F(x(j))) > evalf(F(x(j + 1))) then
                for k from 1 to N fo
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


[Назад на головну](../README.md)
