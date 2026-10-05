"""
Лабораторная работа №2.
Вариант 10. Составление плана дежурств на неделю с ограничениями по сменам.
Группа Б24-517, номер в группе 28.

Запуск: python main.py
Результаты сохраняются в папку lab2_out/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple

# ---------- Настройки ----------
BASE_SEED = 42
N_RUNS = 20

N_EMP = 25            # сотрудники (>= 20 по требованию)
N_DAYS = 7            # горизонт планирования
N_SHIFTS = 3          # утро / день / ночь
N_SLOTS = N_DAYS * N_SHIFTS   # 21 слот (>= 20 по требованию)
MIN_PER_SHIFT = 3     # минимум на смене
MAX_SHIFTS_PER_EMP = 5
SENIOR_IDS = [0, 5, 10, 15, 20]   # 5 старших (могут на ночь)

SHIFT_NAMES = ["Утро", "День", "Ночь"]
DAY_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

def slot_to_day_shift(j: int) -> Tuple[int, int]:
    return j // N_SHIFTS, j % N_SHIFTS

def generate_data(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    prefs = rng.uniform(0.5, 1.5, size=(N_EMP, N_SHIFTS))
    return {"prefs": prefs}

# ---------- Фитнес-функция ----------
def compute_fitness(X: np.ndarray, prefs: np.ndarray,
                    hard_penalty: float = 1000.0,
                    soft_penalty: float = 10.0) -> Tuple[float, dict]:
    violations = {}
    penalty = 0.0

    # 1. Минимум на смене
    per_slot = X.sum(axis=0)
    v1 = np.maximum(MIN_PER_SHIFT - per_slot, 0).sum()
    violations["min_per_shift"] = int(v1)
    penalty += v1 * hard_penalty

    # 2. Максимум смен на сотрудника
    per_emp = X.sum(axis=1)
    v2 = np.maximum(per_emp - MAX_SHIFTS_PER_EMP, 0).sum()
    violations["max_shifts"] = int(v2)
    penalty += v2 * hard_penalty

    # 3. Запрет ночь -> утро
    v3 = 0
    for d in range(N_DAYS - 1):
        night = X[:, d * N_SHIFTS + 2]
        morn = X[:, (d + 1) * N_SHIFTS + 0]
        v3 += int(np.sum((night == 1) & (morn == 1)))
    violations["night_morning"] = v3
    penalty += v3 * hard_penalty

    # 4. Минимум 1 выходной
    working_days = (X.reshape(N_EMP, N_DAYS, N_SHIFTS).sum(axis=2) > 0).sum(axis=1)
    v4 = np.maximum(working_days - (N_DAYS - 1), 0).sum()
    violations["no_day_off"] = int(v4)
    penalty += v4 * hard_penalty

    # 5. Квалификация: неквалифицированные не работают ночью
    non_seniors = [i for i in range(N_EMP) if i not in SENIOR_IDS]
    v5 = 0
    for d in range(N_DAYS):
        night_slot = d * N_SHIFTS + 2
        v5 += int(X[non_seniors, night_slot].sum())
    violations["unqualified_night"] = v5
    penalty += v5 * hard_penalty

    # Мягкие штрафы
    load_var = float(per_emp.var())
    violations["load_var"] = load_var
    penalty += load_var * soft_penalty

    prefs_expanded = np.tile(prefs, (1, N_DAYS))  # (25,3) -> (25,21)
    pref_score = float(np.sum(X * prefs_expanded))
    violations["pref_score"] = pref_score
    penalty -= pref_score * 0.5

    return penalty, violations

# ---------- Инициализация ----------
def init_population(cfg, data: dict) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    pop = np.zeros((cfg.pop_size, N_EMP, N_SLOTS), dtype=np.int8)

    for k in range(cfg.pop_size):
        X = pop[k]
        for i in range(N_EMP):
            allowed_shifts = [0, 1] if i not in SENIOR_IDS else [0, 1, 2]
            n_shifts = rng.integers(2, MAX_SHIFTS_PER_EMP + 1)
            chosen_slots = rng.choice(N_SLOTS, size=n_shifts, replace=False)
            for s in chosen_slots:
                _, sh = slot_to_day_shift(s)
                if sh in allowed_shifts:
                    X[i, s] = 1
        
        # Гарантируем 1 выходной
        for i in range(N_EMP):
            wd = (X[i].reshape(N_DAYS, N_SHIFTS).sum(axis=1) > 0)
            if wd.all():
                drop_day = rng.integers(0, N_DAYS)
                X[i, drop_day * N_SHIFTS:(drop_day + 1) * N_SHIFTS] = 0
        pop[k] = X
    return pop

# ---------- Операторы ----------
def crossover_uniform(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(N_EMP) < 0.5
    c1 = np.where(mask[:, None, None], p1, p2)
    c2 = np.where(mask[:, None, None], p2, p1)
    return c1.astype(np.int8), c2.astype(np.int8)

def mutate_reassign(X: np.ndarray, n_mut: int, rng: np.random.Generator) -> np.ndarray:
    Y = X.copy()
    for _ in range(n_mut):
        i = rng.integers(0, N_EMP)
        allowed_shifts = [0, 1] if i not in SENIOR_IDS else [0, 1, 2]
        working = np.where(Y[i] == 1)[0]
        if len(working) > 0:
            drop = rng.choice(working)
            Y[i, drop] = 0
        for _ in range(10):
            add = rng.integers(0, N_SLOTS)
            _, sh = slot_to_day_shift(add)
            if sh in allowed_shifts and Y[i, add] == 0:
                Y[i, add] = 1
                break
    return Y

def repair(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Y = X.copy()
    non_seniors = [i for i in range(N_EMP) if i not in SENIOR_IDS]
    
    # 1. Квалификация
    for d in range(N_DAYS):
        ns = d * N_SHIFTS + 2
        for i in non_seniors:
            Y[i, ns] = 0

    # 2. Запрет ночь->утро
    for d in range(N_DAYS - 1):
        night = d * N_SHIFTS + 2
        morn = (d + 1) * N_SHIFTS + 0
        conflict = np.where((Y[:, night] == 1) & (Y[:, morn] == 1))[0]
        for i in conflict:
            if rng.random() < 0.5: Y[i, night] = 0
            else: Y[i, morn] = 0

    # 3. Максимум смен
    for i in range(N_EMP):
        working = np.where(Y[i] == 1)[0]
        if len(working) > MAX_SHIFTS_PER_EMP:
            drop = rng.choice(working, size=len(working) - MAX_SHIFTS_PER_EMP, replace=False)
            Y[i, drop] = 0

    # 4. Выходной
    for i in range(N_EMP):
        wd = (Y[i].reshape(N_DAYS, N_SHIFTS).sum(axis=1) > 0)
        if wd.all():
            drop_day = rng.integers(0, N_DAYS)
            Y[i, drop_day * N_SHIFTS:(drop_day + 1) * N_SHIFTS] = 0

    # 5. Минимум на смене
    per_slot = Y.sum(axis=0)
    for j in range(N_SLOTS):
        _, sh = slot_to_day_shift(j)
        while per_slot[j] < MIN_PER_SHIFT:
            candidates = [i for i in range(N_EMP) if Y[i, j] == 0 and (i in SENIOR_IDS or sh != 2) and Y[i].sum() < MAX_SHIFTS_PER_EMP]
            if not candidates: break
            i = rng.choice(candidates)
            Y[i, j] = 1
            per_slot[j] += 1
    return Y

# ---------- ГА ----------
@dataclass
class GAConfig:
    pop_size: int = 80
    max_gens: int = 250
    tournament_k: int = 3
    p_crossover: float = 0.85
    p_mutation: float = 0.4
    n_mut_slots: int = 2
    hard_penalty: float = 1000.0
    soft_penalty: float = 10.0
    use_repair: bool = False
    seed: int = BASE_SEED
    tag: str = "default"

def tournament_idx(fitness: np.ndarray, k: int, rng) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return idx[int(np.argmin(fitness[idx]))]

def run_ga(cfg: GAConfig, data: dict) -> dict:
    rng = np.random.default_rng(cfg.seed)
    pop = init_population(cfg, data)
    prefs = data["prefs"]

    fit = np.array([compute_fitness(x, prefs, cfg.hard_penalty, cfg.soft_penalty)[0] for x in pop])
    best_idx = int(np.argmin(fit))
    history_best = [fit[best_idx]]
    archive_best = (fit[best_idx], pop[best_idx].copy())

    for gen in range(1, cfg.max_gens + 1):
        new_pop = []
        order = np.argsort(fit)
        new_pop.append(pop[order[0]].copy())
        new_pop.append(pop[order[1]].copy())

        while len(new_pop) < cfg.pop_size:
            i1 = tournament_idx(fit, cfg.tournament_k, rng)
            i2 = tournament_idx(fit, cfg.tournament_k, rng)
            if rng.random() < cfg.p_crossover:
                c1, c2 = crossover_uniform(pop[i1], pop[i2], rng)
            else:
                c1, c2 = pop[i1].copy(), pop[i2].copy()
            
            if rng.random() < cfg.p_mutation: c1 = mutate_reassign(c1, cfg.n_mut_slots, rng)
            if rng.random() < cfg.p_mutation: c2 = mutate_reassign(c2, cfg.n_mut_slots, rng)
            
            if cfg.use_repair:
                c1 = repair(c1, rng)
                c2 = repair(c2, rng)
                
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size: new_pop.append(c2)

        pop = np.array(new_pop)
        fit = np.array([compute_fitness(x, prefs, cfg.hard_penalty, cfg.soft_penalty)[0] for x in pop])

        gen_best_idx = int(np.argmin(fit))
        history_best.append(fit[gen_best_idx])
        if fit[gen_best_idx] < archive_best[0]:
            archive_best = (fit[gen_best_idx], pop[gen_best_idx].copy())

    _, viol = compute_fitness(archive_best[1], prefs, cfg.hard_penalty, cfg.soft_penalty)
    return {"best_f": archive_best[0], "best_x": archive_best[1], "violations": viol, "history_best": history_best}

# ---------- Случайный допустимый поиск (Baseline) ----------
def random_valid_search(budget: int, seed: int, data: dict) -> dict:
    rng = np.random.default_rng(seed)
    best_f = float('inf')
    best_x = None
    prefs = data["prefs"]
    
    for _ in range(budget):
        X = np.zeros((N_EMP, N_SLOTS), dtype=np.int8)
        for i in range(N_EMP):
            allowed = [0, 1] if i not in SENIOR_IDS else [0, 1, 2]
            n = rng.integers(2, MAX_SHIFTS_PER_EMP + 1)
            slots = rng.choice(N_SLOTS, size=n, replace=False)
            for s in slots:
                _, sh = slot_to_day_shift(s)
                if sh in allowed: X[i, s] = 1
        
        X = repair(X, rng) # Делаем решение строго допустимым
        f, _ = compute_fitness(X, prefs, 1000.0, 10.0)
        if f < best_f:
            best_f = f
            best_x = X.copy()
            
    return {"best_f": best_f, "best_x": best_x}

# ---------- Эксперименты ----------
def run_series(configs: List[GAConfig], n_runs: int = N_RUNS) -> dict:
    data = generate_data(BASE_SEED)
    results = {}
    for cfg in configs:
        bests, hists, viols = [], [], []
        for r in range(n_runs):
            c = GAConfig(**{**cfg.__dict__, "seed": cfg.seed + r})
            res = run_ga(c, data)
            bests.append(res["best_f"])
            hists.append(res["history_best"])
            viols.append(res["violations"])
        results[cfg.tag] = {"bests": np.array(bests), "hists": np.array(hists), "viols": viols, "cfg": cfg}
    return results

def save_csv(results: dict, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "best_f", "feasible", "min_per_shift", "max_shifts", "night_morning", "no_day_off", "unqualified_night", "load_var"])
        for tag, data in results.items():
            b = data["bests"]
            feasible = sum(1 for v in data["viols"] if v["min_per_shift"]==0 and v["max_shifts"]==0 and v["night_morning"]==0 and v["no_day_off"]==0 and v["unqualified_night"]==0)
            w.writerow([tag, "summary", float(b.min()), f"{feasible}/{N_RUNS}", "", "", "", "", "", ""])
            for i, (bv, v) in enumerate(zip(b, data["viols"])):
                is_feas = int(v["min_per_shift"]==0 and v["max_shifts"]==0 and v["night_morning"]==0 and v["no_day_off"]==0 and v["unqualified_night"]==0)
                w.writerow([tag, i, float(bv), is_feas, v["min_per_shift"], v["max_shifts"], v["night_morning"], v["no_day_off"], v["unqualified_night"], f"{v['load_var']:.3f}"])

def plot_convergence(results: dict, path: str):
    plt.figure(figsize=(9, 5))
    for tag, data in results.items():
        h = data["hists"]
        mean, std = h.mean(axis=0), h.std(axis=0)
        gens = np.arange(len(mean))
        plt.plot(gens, mean, label=tag)
        plt.fill_between(gens, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес (меньше — лучше)")
    plt.title("Сходимость ГА (план дежурств)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_schedule(X: np.ndarray, path: str):
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
                ax.text(i, d, "●" if grid[d, i] else "", ha="center", va="center", fontsize=8, color="black")
    axes[-1].set_xlabel("Сотрудник")
    plt.suptitle("Лучшее расписание дежурств", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def print_schedule(X: np.ndarray):
    print("\n=== Лучшее расписание (фрагмент) ===")
    print(f"{'Сотр.':<6}", end="")
    for d in DAY_NAMES:
        for s in SHIFT_NAMES:
            print(f"{d[:2]}_{s[:2]:<3}", end=" ")
        print("| смен")
    for i in range(min(10, N_EMP)): # Печатаем первых 10 для краткости
        senior = "*" if i in SENIOR_IDS else " "
        print(f"{i:<5}{senior}", end=" ")
        n = 0
        for d in range(N_DAYS):
            for s in range(N_SHIFTS):
                j = d * N_SHIFTS + s
                print(f"  {int(X[i,j])}   ", end="")
                n += int(X[i, j])
        print(f"| {n}")
    print("... (полная матрица сохранена в визуализации)")

def main():
    os.makedirs("lab2_out", exist_ok=True)

    configs = [
        GAConfig(pop_size=80, max_gens=250, use_repair=False, hard_penalty=1000.0, tag="penalty_only"),
        GAConfig(pop_size=80, max_gens=250, use_repair=True, hard_penalty=100.0, tag="repair_soft"),
        GAConfig(pop_size=80, max_gens=250, use_repair=True, hard_penalty=1000.0, tag="repair_hard"),
    ]

    print("Запуск серии экспериментов ЛР №2...")
    results = run_series(configs, n_runs=N_RUNS)
    save_csv(results, "lab2_out/lab2_results.csv")
    plot_convergence(results, "lab2_out/lab2_comparison.png")

    # Лучшее расписание
    best_tag = min(results, key=lambda t: results[t]["bests"].min())
    best_run_idx = int(np.argmin(results[best_tag]["bests"]))
    data = generate_data(BASE_SEED)
    cfg_best = GAConfig(**{**results[best_tag]["cfg"].__dict__, "seed": BASE_SEED + best_run_idx})
    res_best = run_ga(cfg_best, data)
    
    plot_schedule(res_best["best_x"], "lab2_out/lab2_schedule_best.png")
    print_schedule(res_best["best_x"])

    # Сравнение со случайным поиском
    budget = configs[0].pop_size * configs[0].max_gens
    rs_bests = []
    for r in range(N_RUNS):
        rs = random_valid_search(budget=budget, seed=BASE_SEED + 2000 + r, data=data)
        rs_bests.append(rs["best_f"])
    
    print("\n=== Сводка ===")
    print(f"{'Конфигурация':<18} {'min':>8} {'mean':>8} {'median':>8} {'допустимых':>12}")
    for tag, d in results.items():
        b = d["bests"]
        feas = sum(1 for v in d["viols"] if v["min_per_shift"]==0 and v["max_shifts"]==0 and v["night_morning"]==0 and v["no_day_off"]==0 and v["unqualified_night"]==0)
        print(f"{tag:<18} {b.min():8.1f} {b.mean():8.1f} {np.median(b):8.1f} {feas}/{N_RUNS}")
    
    rs_arr = np.array(rs_bests)
    print(f"{'random_valid':<18} {rs_arr.min():8.1f} {rs_arr.mean():8.1f} {np.median(rs_arr):8.1f} {'N/A':>12}")

    print("\n✅ Файлы сохранены в lab2_out/")

if __name__ == "__main__":
    main()