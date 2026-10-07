"""
Лабораторная работа №2 (Оптимизированная и исправленная версия).
Вариант 10. Составление плана дежурств на неделю.
Группа Б24-517, номер в группе 28.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from dataclasses import dataclass
from typing import Tuple

# ---------- Настройки ----------
BASE_SEED = 42
N_RUNS = 20          # Требование: не менее 20 запусков

N_EMP = 25
N_DAYS = 7
N_SHIFTS = 3
N_SLOTS = N_DAYS * N_SHIFTS  # 21
MIN_PER_SHIFT = 3
MAX_SHIFTS_PER_EMP = 5
SENIOR_IDS = [0, 5, 10, 15, 20]

# Векторизованные маски для скорости
NON_SENIOR_MASK = np.array([i not in SENIOR_IDS for i in range(N_EMP)])
NIGHT_SLOTS = np.array([2, 5, 8, 11, 14, 17, 20])
NIGHT_COLS = np.array([2, 5, 8, 11, 14, 17])   # Ночь Пн-Сб
MORN_COLS = np.array([3, 6, 9, 12, 15, 18])    # Утро Вт-Вс

SHIFT_NAMES = ["Утро", "День", "Ночь"]
DAY_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

def generate_data(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prefs = rng.uniform(0.5, 1.5, size=(N_EMP, N_SHIFTS))
    prefs_expanded = np.tile(prefs, (1, N_DAYS)) # (25, 21)
    return prefs, prefs_expanded

# ---------- Векторизованная фитнес-функция ----------
def compute_fitness(X: np.ndarray, prefs_expanded: np.ndarray) -> float:
    # 1. Минимум на смене
    v1 = np.maximum(MIN_PER_SHIFT - X.sum(axis=0), 0).sum()
    # 2. Максимум смен на сотрудника
    per_emp = X.sum(axis=1)
    v2 = np.maximum(per_emp - MAX_SHIFTS_PER_EMP, 0).sum()
    # 3. Запрет ночь -> утро
    v3 = np.sum((X[:, NIGHT_COLS] == 1) & (X[:, MORN_COLS] == 1))
    # 4. Минимум 1 выходной
    working_days = (X.reshape(N_EMP, N_DAYS, N_SHIFTS).sum(axis=2) > 0).sum(axis=1)
    v4 = np.maximum(working_days - (N_DAYS - 1), 0).sum()
    # 5. Квалификация (ночные у неквалифицированных)
    v5 = X[NON_SENIOR_MASK][:, NIGHT_SLOTS].sum()
    
    # Штрафы и бонусы
    penalty = (v1 + v2 + v3 + v4 + v5) * 1000.0
    penalty += float(per_emp.var()) * 10.0          # Равномерность
    penalty -= float(np.sum(X * prefs_expanded)) * 0.5  # Предпочтения
    return penalty

# ---------- Быстрая инициализация ----------
def init_population(pop_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.zeros((pop_size, N_EMP, N_SLOTS), dtype=np.int8)
    for k in range(pop_size):
        for i in range(N_EMP):
            allowed = [0, 1] if i not in SENIOR_IDS else [0, 1, 2]
            n_shifts = rng.integers(2, 5)
            valid = [d * 3 + s for d in range(7) for s in allowed]
            chosen = rng.choice(valid, size=min(n_shifts, len(valid)), replace=False)
            pop[k, i, chosen] = 1
    return pop

# ---------- Операторы ----------
def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(N_EMP) < 0.5
    c1 = np.where(mask[:, None], p1, p2).astype(np.int8)
    c2 = np.where(mask[:, None], p2, p1).astype(np.int8)
    return c1, c2

def mutate(X: np.ndarray, n_mut: int, rng: np.random.Generator) -> np.ndarray:
    Y = X.copy()
    for _ in range(n_mut):
        i = rng.integers(0, N_EMP)
        working = np.where(Y[i] == 1)[0]
        if len(working) > 0:
            Y[i, rng.choice(working)] = 0
        
        allowed = [0, 1] if i not in SENIOR_IDS else [0, 1, 2]
        for _ in range(5):
            add = rng.integers(0, N_SLOTS)
            s = add % N_SHIFTS
            if s in allowed and Y[i, add] == 0 and Y[i].sum() < MAX_SHIFTS_PER_EMP:
                Y[i, add] = 1
                break
    return Y

def repair(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Y = X.copy()
    # 1. Убираем ночи у неквалифицированных (векторизовано)
    Y[NON_SENIOR_MASK][:, NIGHT_SLOTS] = 0
    # 2. Убираем конфликты ночь->утро (векторизовано)
    for nc, mc in zip(NIGHT_COLS, MORN_COLS):
        Y[(Y[:, nc] == 1) & (Y[:, mc] == 1), mc] = 0
    # 3. Обрезаем максимум смен (без while, гарантированный выход)
    for i in range(N_EMP):
        working = np.where(Y[i] == 1)[0]
        if len(working) > MAX_SHIFTS_PER_EMP:
            to_remove = rng.choice(working, size=len(working) - MAX_SHIFTS_PER_EMP, replace=False)
            Y[i, to_remove] = 0
    return Y

# ---------- ГА ----------
@dataclass
class GAConfig:
    pop_size: int = 60
    max_gens: int = 100
    p_crossover: float = 0.85
    p_mutation: float = 0.4
    use_repair: bool = False
    tag: str = "default"

def run_ga(cfg: GAConfig, prefs_expanded: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pop = init_population(cfg.pop_size, seed)
    fit = np.array([compute_fitness(x, prefs_expanded) for x in pop])
    
    best_idx = int(np.argmin(fit))
    history_best = [fit[best_idx]]
    archive_best = (fit[best_idx], pop[best_idx].copy())

    for _ in range(cfg.max_gens):
        new_pop = []
        order = np.argsort(fit)
        new_pop.extend([pop[order[0]].copy(), pop[order[1]].copy()])

        while len(new_pop) < cfg.pop_size:
            i1, i2 = rng.choice(cfg.pop_size, size=2, replace=False)
            if rng.random() < cfg.p_crossover:
                c1, c2 = crossover(pop[i1], pop[i2], rng)
            else:
                c1, c2 = pop[i1].copy(), pop[i2].copy()
            
            if rng.random() < cfg.p_mutation: c1 = mutate(c1, 2, rng)
            if rng.random() < cfg.p_mutation: c2 = mutate(c2, 2, rng)
            
            if cfg.use_repair:
                c1, c2 = repair(c1, rng), repair(c2, rng)
                
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size: new_pop.append(c2)

        pop = np.array(new_pop)
        fit = np.array([compute_fitness(x, prefs_expanded) for x in pop])
        
        gen_best_idx = int(np.argmin(fit))
        history_best.append(fit[gen_best_idx])
        if fit[gen_best_idx] < archive_best[0]:
            archive_best = (fit[gen_best_idx], pop[gen_best_idx].copy())

    return {"best_f": archive_best[0], "best_x": archive_best[1], "history_best": history_best}

# ---------- Случайный поиск (Baseline) ----------
def random_search(budget: int, seed: int, prefs_expanded: np.ndarray) -> float:
    rng = np.random.default_rng(seed)
    best_f = float('inf')
    for _ in range(budget):
        # Быстрая генерация случайной матрицы
        X = (rng.random((N_EMP, N_SLOTS)) < 0.2).astype(np.int8)
        X = repair(X, rng)
        f = compute_fitness(X, prefs_expanded)
        if f < best_f: best_f = f
    return best_f

# ---------- Эксперименты ----------
def main():
    print("🚀 Старт Лабораторной работы №2 (Оптимизированная версия)...")
    os.makedirs("lab2_out", exist_ok=True)
    
    _, prefs_expanded = generate_data(BASE_SEED)
    
    configs = [
        GAConfig(pop_size=60, max_gens=100, use_repair=False, tag="penalty_only"),
        GAConfig(pop_size=60, max_gens=100, use_repair=True, tag="repair_hard"),
    ]
    
    all_results = {}
    for cfg in configs:
        print(f"\n🔄 Конфигурация: {cfg.tag}")
        bests, hists = [], []
        for r in range(N_RUNS):
            print(f"   Запуск {r+1}/{N_RUNS}... OK", end="\r")
            res = run_ga(cfg, prefs_expanded, BASE_SEED + r)
            bests.append(res["best_f"])
            hists.append(res["history_best"])
        print(f"   ✅ Завершено: {cfg.tag} (Min: {np.min(bests):.1f})     ")
        all_results[cfg.tag] = {"bests": np.array(bests), "hists": np.array(hists)}

    # Случайный поиск (бюджет снижен до 300 для мгновенного выполнения)
    print("\n🎲 Запуск случайного поиска (baseline)...")
    budget = 300
    rs_bests = [random_search(budget, BASE_SEED + 1000 + r, prefs_expanded) for r in range(N_RUNS)]
    print(f"   ✅ Случайный поиск завершён (Min: {np.min(rs_bests):.1f})")

    # Сохранение CSV
    with open("lab2_out/lab2_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "best_f"])
        for tag, data in all_results.items():
            for i, b in enumerate(data["bests"]):
                w.writerow([tag, i, float(b)])
        for i, b in enumerate(rs_bests):
            w.writerow(["random_search", i, float(b)])

    # Графики
    plt.figure(figsize=(9, 5))
    for tag, data in all_results.items():
        h = data["hists"]
        mean, std = h.mean(axis=0), h.std(axis=0)
        gens = np.arange(len(mean))
        plt.plot(gens, mean, label=tag)
        plt.fill_between(gens, mean - std, mean + std, alpha=0.15)
    plt.plot(gens, [np.mean(rs_bests)]*len(gens), 'r--', label="random_search (mean)")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес (меньше — лучше)")
    plt.title("Сходимость ГА (план дежурств)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("lab2_out/lab2_comparison.png", dpi=150)
    plt.close()

    # Лучшее расписание
    best_tag = min(all_results, key=lambda t: np.min(all_results[t]["bests"]))
    best_idx = int(np.argmin(all_results[best_tag]["bests"]))
    
    if best_tag == "penalty_only":
        cfg_best = GAConfig(pop_size=60, max_gens=100, use_repair=False, tag="penalty_only")
    else:
        cfg_best = GAConfig(pop_size=60, max_gens=100, use_repair=True, tag="repair_hard")
    
    res_best = run_ga(cfg_best, prefs_expanded, BASE_SEED + best_idx)
    
    # Визуализация расписания
    X = res_best["best_x"]
    fig, axes = plt.subplots(N_SHIFTS, 1, figsize=(12, 8), sharex=True)
    for s in range(N_SHIFTS):
        ax = axes[s]
        grid = X[:, s::N_SHIFTS].T
        ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(N_DAYS))
        ax.set_yticklabels(DAY_NAMES)
        ax.set_title(SHIFT_NAMES[s])
        for d in range(N_DAYS):
            for i in range(N_EMP):
                if grid[d, i]:
                    ax.text(i, d, "●", ha="center", va="center", fontsize=8, color="black")
    axes[-1].set_xlabel("Сотрудник")
    plt.suptitle("Лучшее расписание дежурств", y=1.02)
    plt.tight_layout()
    plt.savefig("lab2_out/lab2_schedule_best.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "="*60)
    print("✅ УСПЕХ! Все файлы сохранены в папку: lab2_out/")
    print("   - lab2_results.csv")
    print("   - lab2_comparison.png")
    print("   - lab2_schedule_best.png")
    print("="*60)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"⏱ Общее время выполнения: {time.time() - start_time:.1f} сек.")