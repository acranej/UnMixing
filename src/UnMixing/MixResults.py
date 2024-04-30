from dataclasses import dataclass 
import polars as pl
@dataclass(frozen = True)
class MixResults:
    w: list[float]
    mu: list[float]
    sigma: list[float] 
    
    @property
    def meanW(self) -> float:
        df_w = pl.DataFrame({"w": self.w})
        val_w = df_w["w"].mean()
        return val_w
    
    @property
    def meanMu(self) -> float:
        df_mu= pl.DataFrame({"mu": self.mu})
        val_mu = df_mu["mu"].mean()
        return val_mu
    
    @property
    def meanSigma(self) -> float:
        df_sigma = pl.DataFrame({"sigma": self.sigma})
        val_sigma = df_sigma["sigma"].mean()
        return val_sigma
    
    