
from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import anthropic
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TipoAmparo(str, Enum):
    directo = "directo"
    indirecto = "indirecto"
    no_identificado = "no_identificado"


class Materia(str, Enum):
    civil = "civil"
    penal = "penal"
    administrativa = "administrativa"
    laboral = "laboral"
    mixta = "mixta"
    no_identificada = "no_identificada"


class TipoAutoridad(str, Enum):
    ordenadora = "ordenadora"
    ejecutora = "ejecutora"
    no_especificado = "no_especificado"


class TipoPrueba(str, Enum):
    documental = "documental"
    testimonial = "testimonial"
    pericial = "pericial"
    presuncional = "presuncional"
    otro = "otro"


class TipoSuspension(str, Enum):
    provisional = "provisional"
    definitiva = "definitiva"
    no_aplica = "no_aplica"


class ConfianzaExtraccion(str, Enum):
    alta = "alta"
    media = "media"
    baja = "baja"


# ─────────────────────────────────────────────
# Modelos Pydantic
# ─────────────────────────────────────────────

class Quejoso(BaseModel):
    nombre: Optional[str] = None
    domicilio: Optional[str] = None
    representante_legal: Optional[str] = None


class AutoridadResponsable(BaseModel):
    nombre: str
    tipo: TipoAutoridad = TipoAutoridad.no_especificado
    acto_atribuido: Optional[str] = None


class TerceroInteresado(BaseModel):
    nombre: str
    domicilio: Optional[str] = None


class MinisterioPublico(BaseModel):
    presente: bool = False
    adscripcion: Optional[str] = None


class ActoReclamado(BaseModel):
    descripcion: Optional[str] = None
    fecha_acto: Optional[str] = Field(
        None,
        description="Formato ISO 8601: YYYY-MM-DD. Si solo hay año y mes, usar YYYY-MM-01."
    )
    norma_general_impugnada: Optional[str] = None


class DerechoViolado(BaseModel):
    derecho: str
    articulo_constitucional: Optional[str] = None
    articulo_convencional: Optional[str] = None


class ConceptoViolacion(BaseModel):
    numero: int
    resumen: str = Field(..., description="Máximo 80 palabras.")
    argumento_central: str = Field(..., description="Máximo 40 palabras.")


class PruebaOfrecida(BaseModel):
    tipo: TipoPrueba
    descripcion: str


class Suspension(BaseModel):
    solicitada: bool = False
    tipo: TipoSuspension = TipoSuspension.no_aplica


class ResultadoAmparo(BaseModel):
    """Schema completo de extracción de un juicio de amparo mexicano."""

    tipo_amparo: TipoAmparo = TipoAmparo.no_identificado
    numero_expediente: Optional[str] = None
    fecha_presentacion: Optional[str] = Field(
        None,
        description="Formato ISO 8601: YYYY-MM-DD."
    )
    juzgado_o_tribunal: Optional[str] = None
    materia: Materia = Materia.no_identificada

    quejoso: Quejoso = Field(default_factory=Quejoso)
    autoridades_responsables: list[AutoridadResponsable] = Field(default_factory=list)
    terceros_interesados: list[TerceroInteresado] = Field(default_factory=list)
    ministerio_publico: MinisterioPublico = Field(default_factory=MinisterioPublico)
    acto_reclamado: ActoReclamado = Field(default_factory=ActoReclamado)
    derechos_violados: list[DerechoViolado] = Field(default_factory=list)
    conceptos_de_violacion: list[ConceptoViolacion] = Field(default_factory=list)
    pruebas_ofrecidas: list[PruebaOfrecida] = Field(default_factory=list)
    suspension: Suspension = Field(default_factory=Suspension)

    confianza_extraccion: ConfianzaExtraccion = ConfianzaExtraccion.media
    observaciones: Optional[str] = None