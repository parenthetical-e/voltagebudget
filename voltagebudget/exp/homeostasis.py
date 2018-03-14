import numpy as np

from voltagebudget.exp.autotune import autotune_homeostasis
from voltagebudget.exp import sweep_A


def homeostasis(name,
                stim,
                E_0,
                target=0,
                N=250,
                d=-2e-3,
                w=2e-3,
                t=0.4,
                A=0.05e-9,
                Z_0=1e-6,
                Z_max=1,
                f=8,
                T=0.125,
                n_jobs=1,
                mode='regular',
                noise=False,
                no_lock=False,
                verbose=False,
                save_only=False,
                save_details=False,
                seed_value=42):

    # Find best Z
    Z_hat, fopt = autotune_homeostasis(
        stim,
        target,
        E_0=E_0,
        N=N,
        t=t,
        A=A,
        Z_0=Z_0,
        Z_max=Z_max,
        f=f,
        n_jobs=n_jobs,
        mode=mode,
        noise=noise,
        no_lock=no_lock,
        verbose=verbose,
        seed_value=42)

    if verbose:
        print(">>> Z_hat {}, fopt {}".format(Z_hat, fopt))

    # Use sweep_A to analyze Z_hat 
    sweep_A(
        name,
        stim,
        E_0,
        A_0=A,
        A_max=A,
        Z=Z_hat,
        n_samples=1,
        d=d,
        w=w,
        T=1 / f,
        f=f,
        N=N,
        mode=mode,
        noise=noise,
        no_lock=no_lock,
        verbose=verbose,
        save_only=save_only,
        save_details=save_details,
        seed_value=seed_value)
