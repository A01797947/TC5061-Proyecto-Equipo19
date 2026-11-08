"""Contratos HTTP de entrada y salida de MiBici.

Los nombres de P1 y P2 reproducen las columnas reales del entrenamiento.
P3 usa un mapa de variables porque el ganador puede ser baseline, candidato,
perfil o híbrido; el servicio valida ese mapa con la firma MLflow exportada.
"""
from __future__ import annotations

from datetime import date
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class P1Request(StrictInput):
    Año: int = Field(ge=2014, le=2100)
    Mes: int = Field(ge=1, le=12)
    Dia: int = Field(ge=1, le=31)
    Dia_semana: int = Field(ge=0, le=6, description="Lunes=0, domingo=6")
    origin_station_id: str | int = Field(description="Identificador de estación de origen")

    @field_validator("origin_station_id")
    @classmethod
    def valid_station_id(cls, value: str | int) -> str | int:
        if isinstance(value, bool) or not str(value).strip() or len(str(value)) > 50:
            raise ValueError("Identificador de estación inválido")
        return value

    @model_validator(mode="after")
    def validar_calendario(self):
        fecha = date(self.Año, self.Mes, self.Dia)
        if fecha.weekday() != self.Dia_semana:
            raise ValueError("Dia_semana no coincide con la fecha; lunes=0")
        return self


class P2Request(StrictInput):
    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    fin_de_semana: int = Field(ge=0, le=1)
    feriado: int = Field(ge=0, le=1)
    mes: int = Field(ge=1, le=12)
    hora_sin: FiniteFloat = Field(ge=-1, le=1)
    hora_cos: FiniteFloat = Field(ge=-1, le=1)
    capacidad: FiniteFloat = Field(gt=0)
    latitud: FiniteFloat = Field(ge=-90, le=90)
    longitud: FiniteFloat = Field(ge=-180, le=180)
    ocupacion_est_h: FiniteFloat = Field(ge=0, le=1)
    neto_h: FiniteFloat
    neto_h_1: FiniteFloat
    neto_ult_3h: FiniteFloat
    salidas_h: FiniteFloat = Field(ge=0)
    llegadas_h: FiniteFloat = Field(ge=0)
    neto_ventana_ayer: FiniteFloat | None
    neto_ventana_semana_pasada: FiniteFloat | None
    ocup_min_ventana_semana_pasada: FiniteFloat | None = Field(ge=0, le=1)
    ocup_max_ventana_semana_pasada: FiniteFloat | None = Field(ge=0, le=1)
    riesgo_semana_pasada: FiniteFloat | None = Field(ge=0, le=1)
    salidas_sistema_h: FiniteFloat = Field(ge=0)
    actividad_sistema_vs_semana_pasada: FiniteFloat | None

    @model_validator(mode="after")
    def validar_variables_relacionadas(self):
        if self.fin_de_semana != int(self.dia_semana >= 5):
            raise ValueError("fin_de_semana no coincide con dia_semana")
        # Mismo cálculo que features/base.py: sin/cos(2*pi*hora/24).
        angulo = 2 * math.pi * self.hora / 24
        if (abs(self.hora_sin - math.sin(angulo)) > 1e-3
                or abs(self.hora_cos - math.cos(angulo)) > 1e-3):
            raise ValueError(
                f"hora_sin/hora_cos no coinciden con hora={self.hora}; "
                f"esperado {math.sin(angulo):.6f} y {math.cos(angulo):.6f}"
            )
        if (self.ocup_min_ventana_semana_pasada is not None
                and self.ocup_max_ventana_semana_pasada is not None
                and self.ocup_min_ventana_semana_pasada > self.ocup_max_ventana_semana_pasada):
            raise ValueError("ocup_min_ventana_semana_pasada debe ser <= ocup_max_ventana_semana_pasada")
        return self


class P3Request(StrictInput):
    features: dict[str, FiniteFloat] = Field(min_length=1, description="Variables agregadas por estación")


class ModelInfo(BaseModel):
    modelo: str
    version: str
    alias: str
    variables: list[str]


class P1Response(BaseModel):
    modelo: str
    version: str
    prediccion_viajes: float


class P2Response(BaseModel):
    modelo: str
    version: str
    riesgo: int = Field(ge=0, le=1)
    probabilidad_riesgo: float | None = Field(ge=0, le=1)


class P3Response(BaseModel):
    modelo: str
    version: str
    cluster: int = Field(ge=0)
