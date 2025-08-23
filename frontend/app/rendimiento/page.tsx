"use client"
import { useState } from "react"
import type React from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { Brain, Home, Info, ChevronDown, ChevronRight, Code, Camera } from "lucide-react"

// Confusion matrix data for each category - VALORES REALES DEL MODELO
const confusionMatrices = {
  Cardiovascular: { TN: 447, FP: 12, FN: 6, TP: 248, F1: 0.965, P: 0.974, R: 0.961 },
  Hepatorenal: { TN: 487, FP: 8, FN: 15, TP: 203, F1: 0.946, P: 0.984, R: 0.911 },
  Neurological: { TN: 320, FP: 36, FN: 29, TP: 328, F1: 0.910, P: 0.899, R: 0.919 },
  Oncological: { TN: 580, FP: 13, FN: 8, TP: 112, F1: 0.914, P: 0.978, R: 0.933 },
}

const globalMetrics = [
  { name: "Average Precision", spanishName: "Precisión Promedio", value: 0.9589 },
  { name: "F1 Macro", spanishName: "F1 Macro", value: 0.949 },
  { name: "F1 Micro", spanishName: "F1 Micro", value: 0.943 },
  { name: "F1 Weighted", spanishName: "F1 Ponderado", value: 0.939 },
  { name: "Exact Match Ratio", spanishName: "Coincidencia Exacta", value: 0.8808 },
  { name: "Hamming Loss", spanishName: "Pérdida de Hamming", value: 0.0336 },
]

const experimentPrompts = [
  {
    id: 1,
    title: "Primer Prompt",
    content: `Create a modern and minimalistic website with a medical-inspired aesthetic 
(soft colors like light blue, green, light gray, with turquoise accents), 
without being exaggerated, leaning more toward minimalism.

The application should represent an Artificial Intelligence model that, 
given a title and an abstract of a biomedical article, classifies it into 
one of the following classes:
- Cardiovascular
- Hepatorenal
- Neurological
- Oncological

Design requirements:
- A clean, minimalistic style with a clear font (e.g., Inter, Roboto, or Lato).
- A top header with the title: "Clasificador de Artículos Biomédicos con IA"
  and a subtitle: "Clasificación automática de literatura médica en 
  4 dominios especializados".
- A central section with:
    * A form with 2 fields labeled in Spanish: "Título del artículo" 
      and "Resumen del artículo".
    * A stylish button labeled "Diagnosticar con IA" to submit.
    * A results section showing the predicted class in a minimalist 
      colored card:
        Blue  (#3B82F6) → Cardiovascular
        · Green (#10B981) → Hepatorenal
        · Purple(#8B5CF6) → Neurological
        · Red   (#EF4444) → Oncological
- A small informational section at the bottom (in Spanish) explaining 
  briefly how the model works and the importance of automatic classification 
  in medical research.
- Use smooth animations (e.g., button hover, fade-in for results).
- The layout should be responsive for both desktop and mobile.

Suggested color palette:
- Main background: #F9FAFB (very light gray)
- Text: #1F2937 (dark gray)`,
  },
  {
    id: 2,
    title: "Segundo Prompt",
    content: `Add a second page to the website called "Rendimiento del Modelo", focused on visualizing 
the evaluation results of the classifier. This page should include:
A title: "Rendimiento del Clasificador" 
and subtitle: "Evaluación cuantitativa del desempeño en los cuatro dominios médicos".
A grid layout with four
confusion matrices (2x2 small heatmaps), one for each class:
Cardiovascular: TN=451, FP=8, FN=10, TP=244 
(F1=0.964, P=0.968, R=0.961).
Hepatorenal: TN=495, FP=0, FN=17, TP=201 (F1=0.959,
P=1.000, R=0.922).
Neurological: TN=339, FP=17, FN=42, TP=315 (F1=0.914, P=0.949, R=0.882).
Oncological: TN=586, FP=7, FN=7, TP=113 (F1=0.942, P=0.942, R=0.942).

Each matrix should be shown with interactive tooltips (hover to see counts), 
color intensity 
according to value, and a small caption in Spanish (ej. "Matriz de confusión 

Cardiovascular").
A bar chart comparing global performance metrics:F1 Micro = 0.9430F1 Macro = 0.9463
F1 Weighted = 0.9428
Hamming Loss = 0.0372Average Precision = 0.9753Exact Match Ratio = 0.8668Each bar labeled clearly with its value, and a short caption in Spanish: "Métricas globales de evaluación". 
At the bottom, add a small explanatory text box in Spanish summarizing: 
"El modelo presenta un desempeño robusto en los cuatro dominios médicos, con valores 
de F1 superiores al 0.91 en todas las clases y un Hamming Loss de 0.0372. 
La exactitud promedio y la cobertura de etiquetas muestran que el sistema es confiable 
para la clasificación automática de artículos biomédicos.
"Maintain the same minimalistic, medical-inspired style as the homepage 
(light backgrounds, turquoise accents, clean typography, smooth hover effects). Ensure 
the layout is responsive for desktop and mobile.`,
  },
  {
    id: 3,
    title: "Tercer Prompt",
    content: `In the section "Métricas Globales de Evaluación", instead of showing the
values as progress bars or text, create a horizontal bar plot where each bar represents 
a global metric of the model.Metrics to include:F1 Micro = 0.9430F1 Macro = 0.9463F1 
Weighted = 0.9428Hamming Loss = 0.0372Average 
Precision = 0.9753Exact Match Ratio = 0.8668 
Use one bar per metric, with the value shown at the end of the bar.
Order the bars from highest to lowest.
Add labels in Spanish on the left (ej. "F1 Micro", "F1 Macro", "Hamming Loss").
Style: minimalistic, clean, with turquoise accent color for the bars.
Add a short caption in Spanish below: 
"Visualización comparativa de métricas globales del clasificador".
Ensure the bar plot is interactive (hover shows exact value) and responsive
for desktop and mobile.`,
  },
]

const experimentScreenshots = [
  {
    id: 1,
    title: "Curvas ROC por Categoría",
    description: "Curvas ROC individuales para cada dominio médico",
    thumbnail: "/evidencias/curva_roc.png",
    fullSize: "/evidencias/curva_roc.png",
  },
  {
    id: 2,
    title: "Notebook Jupyter - Análisis 1",
    description: "Experimentación y desarrollo del modelo en Jupyter",
    thumbnail: "/evidencias/jupyter1.png",
    fullSize: "/evidencias/jupyter1.png",
  },
  {
    id: 3,
    title: "Notebook Jupyter - Análisis 2",
    description: "Evaluación y métricas del modelo en Jupyter",
    thumbnail: "/evidencias/jupyter2.png",
    fullSize: "/evidencias/jupyter2.png",
  },
  {
    id: 4,
    title: "Prompt Engineering",
    description: "Proceso de desarrollo con prompts para la aplicación",
    thumbnail: "/evidencias/prompt1.png",
    fullSize: "/evidencias/prompt1.png",
  },
]

const getCategoryColor = (category: string) => {
  switch (category) {
    case "Cardiovascular":
      return "text-blue-600 border-blue-200 bg-blue-50"
    case "Hepatorenal":
      return "text-green-600 border-green-200 bg-green-50"
    case "Neurological":
      return "text-purple-600 border-purple-200 bg-purple-50"
    case "Oncological":
      return "text-red-600 border-red-200 bg-red-50"
    default:
      return "text-gray-600 border-gray-200 bg-gray-50"
  }
}

const getIntensityColor = (value: number, max: number) => {
  const intensity = value / max
  if (intensity > 0.8) return "bg-primary text-primary-foreground"
  if (intensity > 0.6) return "bg-primary/80 text-primary-foreground"
  if (intensity > 0.4) return "bg-primary/60 text-foreground"
  if (intensity > 0.2) return "bg-primary/40 text-foreground"
  return "bg-primary/20 text-foreground"
}

function ConfusionMatrix({ category, data }: { category: string; data: any }) {
  const [hoveredCell, setHoveredCell] = useState<string | null>(null)
  const maxValue = Math.max(data.TN, data.FP, data.FN, data.TP)

  return (
    <Card className={`shadow-lg border-2 ${getCategoryColor(category)} transition-all duration-200 hover:shadow-xl`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${
              category === "Cardiovascular"
                ? "bg-blue-500"
                : category === "Hepatorenal"
                  ? "bg-green-500"
                  : category === "Neurological"
                    ? "bg-purple-500"
                    : "bg-red-500"
            }`}
          ></div>
          {category}
        </CardTitle>
        <CardDescription className="text-sm">Matriz de confusión – {category}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-1 mb-4 relative">
          {/* Matrix cells */}
          <div
            className={`aspect-square flex items-center justify-center text-sm font-semibold rounded cursor-pointer transition-all duration-200 ${getIntensityColor(data.TN, maxValue)}`}
            onMouseEnter={() => setHoveredCell("TN")}
            onMouseLeave={() => setHoveredCell(null)}
          >
            {data.TN}
            {hoveredCell === "TN" && (
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                TN: {data.TN} (Verdaderos Negativos)
              </div>
            )}
          </div>
          <div
            className={`aspect-square flex items-center justify-center text-sm font-semibold rounded cursor-pointer transition-all duration-200 ${getIntensityColor(data.FP, maxValue)}`}
            onMouseEnter={() => setHoveredCell("FP")}
            onMouseLeave={() => setHoveredCell(null)}
          >
            {data.FP}
            {hoveredCell === "FP" && (
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                FP: {data.FP} (Falsos Positivos)
              </div>
            )}
          </div>
          <div
            className={`aspect-square flex items-center justify-center text-sm font-semibold rounded cursor-pointer transition-all duration-200 ${getIntensityColor(data.FN, maxValue)}`}
            onMouseEnter={() => setHoveredCell("FN")}
            onMouseLeave={() => setHoveredCell(null)}
          >
            {data.FN}
            {hoveredCell === "FN" && (
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                FN: {data.FN} (Falsos Negativos)
              </div>
            )}
          </div>
          <div
            className={`aspect-square flex items-center justify-center text-sm font-semibold rounded cursor-pointer transition-all duration-200 ${getIntensityColor(data.TP, maxValue)}`}
            onMouseEnter={() => setHoveredCell("TP")}
            onMouseLeave={() => setHoveredCell(null)}
          >
            {data.TP}
            {hoveredCell === "TP" && (
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                TP: {data.TP} (Verdaderos Positivos)
              </div>
            )}
          </div>
        </div>

        {/* Metrics */}
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">F1:</span>
            <span className="font-semibold">{data.F1.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Precisión:</span>
            <span className="font-semibold">{data.P.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Recall:</span>
            <span className="font-semibold">{data.R.toFixed(3)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function HorizontalBarChart() {
  const [hoveredBar, setHoveredBar] = useState<string | null>(null)

  return (
    <div className="space-y-4">
      {globalMetrics.map((metric, index) => {
        // For Hamming Loss, we want to show it as an inverse (lower is better)
        const displayValue = metric.name === "Hamming Loss" ? 1 - metric.value : metric.value
        const barWidth = metric.name === "Hamming Loss" ? (1 - metric.value) * 100 : metric.value * 100

        return (
          <div
            key={metric.name}
            className="relative"
            onMouseEnter={() => setHoveredBar(metric.name)}
            onMouseLeave={() => setHoveredBar(null)}
          >
            <div className="flex items-center gap-4 mb-2">
              <div className="w-32 text-sm font-medium text-muted-foreground text-right">{metric.spanishName}</div>
              <div className="flex-1 relative">
                <div className="w-full bg-muted rounded-full h-6 relative overflow-hidden">
                  <div
                    className="bg-primary h-6 rounded-full transition-all duration-700 ease-out flex items-center justify-end pr-3"
                    style={{
                      width: `${barWidth}%`,
                      animationDelay: `${index * 150}ms`,
                    }}
                  >
                    <span className="text-xs font-semibold text-primary-foreground">{metric.value.toFixed(4)}</span>
                  </div>
                </div>

                {/* Hover tooltip */}
                {hoveredBar === metric.name && (
                  <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-black text-white px-3 py-1 rounded text-xs whitespace-nowrap z-10">
                    {metric.spanishName}: {metric.value.toFixed(4)}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}

      {/* Caption */}
      <p className="text-sm text-muted-foreground text-center mt-6 italic">
        Visualización comparativa de métricas globales del clasificador
      </p>
    </div>
  )
}

function CollapsibleSection({
  title,
  icon: Icon,
  children,
  defaultOpen = false,
}: {
  title: string
  icon: any
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
      <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => setIsOpen(!isOpen)}>
        <CardTitle className="flex items-center justify-between text-lg">
          <div className="flex items-center gap-2">
            <Icon className="h-5 w-5 text-primary" />
            {title}
          </div>
          {isOpen ? (
            <ChevronDown className="h-5 w-5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-5 w-5 text-muted-foreground" />
          )}
        </CardTitle>
      </CardHeader>
      {isOpen && <CardContent className="pt-0">{children}</CardContent>}
    </Card>
  )
}

function ScreenshotGallery() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {experimentScreenshots.map((screenshot) => (
          <div
            key={screenshot.id}
            className="group cursor-pointer"
            onClick={() => setSelectedImage(screenshot.fullSize)}
          >
            <div className="relative overflow-hidden rounded-lg border border-border hover:border-primary/50 transition-all duration-200">
              <img
                src={screenshot.thumbnail || "/placeholder.svg"}
                alt={screenshot.title}
                className="w-full h-32 object-cover group-hover:scale-105 transition-transform duration-200"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-200" />
            </div>
            <div className="mt-2">
              <h4 className="font-medium text-sm">{screenshot.title}</h4>
              <p className="text-xs text-muted-foreground">{screenshot.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for expanded image */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img
              src={selectedImage || "/placeholder.svg"}
              alt="Expanded screenshot"
              className="max-w-full max-h-full object-contain rounded-lg"
            />
            <Button
              variant="outline"
              size="sm"
              className="absolute top-4 right-4 bg-background/90"
              onClick={() => setSelectedImage(null)}
            >
              ✕
            </Button>
          </div>
        </div>
      )}
    </>
  )
}

export default function PerformancePage() {
  const [activeTab, setActiveTab] = useState<"performance" | "evidence">("performance")
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-3xl font-bold text-foreground">
                  {activeTab === "performance" ? "Rendimiento del Clasificador" : "Evidencia Experimental"}
                </h1>
              </div>
              <p className="text-muted-foreground text-lg">
                {activeTab === "performance"
                  ? "Evaluación cuantitativa del desempeño en los cuatro dominios médicos"
                  : "Evidencia recopilada durante el proceso de experimentación"}
              </p>
            </div>
            <Link href="/">
              <Button
                variant="outline"
                className="flex items-center gap-2 hover:bg-primary/10 transition-colors bg-transparent"
              >
                <Home className="h-4 w-4" />
                Volver al Clasificador
              </Button>
            </Link>
          </div>

          {/* Tab navigation */}
          <div className="flex gap-2 mt-6">
            <Button
              variant={activeTab === "performance" ? "default" : "outline"}
              onClick={() => setActiveTab("performance")}
              className="flex items-center gap-2"
            >
              <Info className="h-4 w-4" />
              Rendimiento
            </Button>
            <Button
              variant={activeTab === "evidence" ? "default" : "outline"}
              onClick={() => setActiveTab("evidence")}
              className="flex items-center gap-2"
            >
              <Code className="h-4 w-4" />
              Evidencia
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto space-y-8">
          {activeTab === "performance" ? (
            <>
              {/* Confusion Matrices Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {Object.entries(confusionMatrices).map(([category, data]) => (
                  <ConfusionMatrix key={category} category={category} data={data} />
                ))}
              </div>

              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-xl">Métricas Globales de Evaluación</CardTitle>
                  <CardDescription>Rendimiento general del modelo en todos los dominios médicos</CardDescription>
                </CardHeader>
                <CardContent>
                  <HorizontalBarChart />
                </CardContent>
              </Card>

              {/* Explanatory Text */}
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Info className="h-5 w-5 text-primary" />
                    Resumen del Rendimiento
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">
                    El modelo presenta un desempeño robusto en los cuatro dominios médicos, con valores de F1 superiores
                    al 0.91 en todas las clases y un Hamming Loss de 0.0336. El threshold de clasificación está configurado
                    en 0.36 (36%), optimizado para maximizar el F1 Score Weighted. La exactitud promedio y la cobertura de
                    etiquetas muestran que el sistema es confiable para la clasificación automática de artículos
                    biomédicos.
                  </p>

                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-green-50 border border-green-200">
                      <h4 className="font-semibold text-green-800 mb-2">Fortalezas del Modelo</h4>
                      <ul className="text-sm text-green-700 space-y-1">
                        <li>• Oncológico: Precisión más alta (0.978)</li>
                        <li>• Cardiovascular: F1 más alto (0.965)</li>
                        <li>• Threshold optimizado: 0.36 (36%)</li>
                      </ul>
                    </div>

                    <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                      <h4 className="font-semibold text-blue-800 mb-2">Métricas Destacadas</h4>
                      <ul className="text-sm text-blue-700 space-y-1">
                        <li>• F1 Macro: 0.949 </li>
                        <li>• Average Precision: 0.9589</li>
                        <li>• Exact Match: 88.08% de casos perfectos</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <>
              {/* Evidence tab content */}
              <div className="space-y-6">
                <CollapsibleSection title="Prompts Utilizados" icon={Code} defaultOpen={true}>
                  <div className="space-y-4">
                    {experimentPrompts.map((prompt) => (
                      <div key={prompt.id} className="border border-border rounded-lg p-4">
                        <h4 className="font-semibold mb-3 text-foreground">{prompt.title}</h4>
                        <pre className="bg-muted p-4 rounded-md text-sm overflow-x-auto whitespace-pre-wrap font-mono text-muted-foreground">
                          {prompt.content}
                        </pre>
                      </div>
                    ))}
                  </div>
                </CollapsibleSection>

                <CollapsibleSection title="Capturas de Pantalla" icon={Camera}>
                  <ScreenshotGallery />
                </CollapsibleSection>

                <p className="text-sm text-muted-foreground text-center italic">
                  Evidencia recopilada durante el proceso de experimentación.
                </p>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
