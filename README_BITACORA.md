# Documentación del Sistema de Cifrado

Este directorio contiene la documentación completa del proceso de cifrado de imágenes implementado en `Maestro_TLS_Keystream_XYZ.py`.

## Archivos Disponibles

### 1. Bitácora en Markdown
- **Archivo:** `BITACORA_PROCESO_CIFRADO.md`
- **Tamaño:** 29 KB
- **Formato:** Markdown
- **Uso:** Lectura rápida, edición simple, visualización en GitHub

### 2. Bitácora en LaTeX
- **Archivo:** `BITACORA_PROCESO_CIFRADO.tex`
- **Tamaño:** 33 KB
- **Formato:** LaTeX source
- **Uso:** Edición avanzada, personalización profesional

### 3. Bitácora en PDF
- **Archivo:** `BITACORA_PROCESO_CIFRADO.pdf`
- **Tamaño:** 339 KB
- **Páginas:** 20
- **Formato:** PDF
- **Uso:** Presentación profesional, impresión, distribución

## Contenido de la Bitácora

La documentación incluye:

1. **Resumen Ejecutivo**
2. **Fase de Inicialización y Configuración**
3. **Fase de Carga y Vectorización de la Imagen**
4. **Fase de Difusión (Mapa Logístico)**
5. **Fase de Confusión (Oscilador de Rössler)**
6. **Fase de Preparación de Datos**
7. **Fase de Transmisión MQTT con TLS**
8. **Fase de Generación de Métricas y Gráficas**
9. **Resumen del Flujo Completo**
10. **Análisis de Ejemplo Completo**
11. **Conclusiones**
12. **Glosario**

### Elementos Destacados

- ✓ **12 tablas** con datos de ejemplo
- ✓ **Ecuaciones matemáticas** formateadas
- ✓ **Código fuente** con resaltado de sintaxis
- ✓ **Diagramas de flujo** textuales
- ✓ **Análisis de seguridad** detallado

## Compilación del Documento LaTeX

Para recompilar el PDF desde el código fuente LaTeX:

### Requisitos
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-lang-spanish texlive-fonts-recommended
```

### Compilación
```bash
# Primera compilación
pdflatex BITACORA_PROCESO_CIFRADO.tex

# Segunda compilación (para referencias cruzadas y tabla de contenidos)
pdflatex BITACORA_PROCESO_CIFRADO.tex
```

El proceso generará:
- `BITACORA_PROCESO_CIFRADO.pdf` - Documento final
- Archivos auxiliares (*.aux, *.log, *.out, *.toc) - Ignorados por .gitignore

## Estructura del Documento LaTeX

El documento utiliza los siguientes paquetes principales:

- **babel (spanish)**: Soporte de idioma español
- **geometry**: Márgenes de página personalizados
- **amsmath, amssymb**: Ecuaciones matemáticas
- **booktabs, longtable**: Tablas profesionales
- **listings**: Código fuente con resaltado
- **hyperref**: Enlaces internos y metadatos PDF
- **fancyhdr**: Encabezados y pies de página personalizados

## Personalización

### Cambiar Márgenes
Editar en el preámbulo:
```latex
\usepackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
```

### Cambiar Fuente
Reemplazar `lmodern` con otra fuente:
```latex
\usepackage{times}  % Times New Roman
% o
\usepackage{palatino}  % Palatino
```

### Añadir Colores a Tablas
Las tablas ya incluyen el paquete `colortbl` para personalización.

## Información Técnica

### Contexto del Sistema
- **Plataforma:** Raspberry Pi 4
- **Imagen de Ejemplo:** RGB 250×250 píxeles (187,500 valores)
- **Técnicas Criptográficas:**
  - Difusión: Mapa Logístico
  - Confusión: Oscilador de Rössler
  - Transmisión: MQTT con TLS

### Métricas del Documento
- Secciones principales: 10
- Subsecciones: 35+
- Tablas: 12
- Ecuaciones: 6
- Bloques de código: 20+

## Licencia y Autoría

- **Autor:** Sistema de Documentación Automática
- **Fecha:** 13 de febrero de 2026
- **Versión:** 1.0
- **Código Analizado:** `Maestro_TLS_Keystream_XYZ.py`

---

Para más información sobre el sistema de cifrado, consulte el código fuente en `Maestro_TLS_Keystream_XYZ.py`.
