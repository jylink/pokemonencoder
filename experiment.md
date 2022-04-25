Setting
* encoder: cifar10
* downstream: stl10
* attack: LinfPGD(epsilon=[0.05, 0.1, 0.2], untargeted and targeted)

| backdoor | trigger              | CA | ASR | untargeted | targeted |
| -------- | :------------------- | :- | --- | ---------- | -------- |
| clean    |                      | 75 |     | 100        | 100      |
| mew      | badencoder           | 76 | 11  | 100        | 100      |
| mew      | trapdoor(alpha=0.1)  | 70 | 16  | 100        | 93       |
| mew      | fullnoise(alpha=0.1) | 71 | 10  | 100        | 82       |
| mew      | fullnoise(alpha=0.5) | 76 | 10  | 100        | 100      |
| ditto    | fullnoise(alpha=0.1) | 74 | 35  | 100        | 100      |
