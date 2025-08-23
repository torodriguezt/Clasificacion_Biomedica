"use client"
import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { Brain, Home, Info } from "lucide-react"

// Confusion matrix data for each category
const confusionMatrices = {
  Cardiovascular: { TN: 451, FP: 8, FN: 10, TP: 244, F1: 0.964, P: 0.968, R: 0.961 },
  Hepatorenal: { TN: 495, FP: 0, FN: 17, TP: 201, F1: 0.959, P: 1.0, R: 0.922 },
  Neurological: { TN: 339, FP: 17, FN: 42, TP: 315, F1: 0.914, P: 0.949, R: 0.882 },
  Oncological: { TN: 586, FP: 7, FN: 7, TP: 113, F1: 0.942, P: 0.942, R: 0.942 },
}

const globalMetrics = [
  { name: "Average Precision", spanishName: "Precisión Promedio", value: 0.9753 },
  { name: "F1 Macro", spanishName: "F1 Macro", value: 0.9463 },
  { name: "F1 Micro", spanishName: "F1 Micro", value: 0.943 },
  { name: "F1 Weighted", spanishName: "F1 Ponderado", value: 0.9428 },
  { name: "Exact Match Ratio", spanishName: "Coincidencia Exacta", value: 0.8668 },
  { name: "Hamming Loss", spanishName: "Pérdida de Hamming", value: 0.0372 },
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

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-3xl font-bold text-foreground">Rendimiento del Clasificador</h1>
              </div>
              <p className="text-muted-foreground text-lg">
                Evaluación cuantitativa del desempeño en los cuatro dominios médicos
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
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto space-y-8">
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
                El modelo presenta un desempeño robusto en los cuatro dominios médicos, con valores de F1 superiores al
                0.91 en todas las clases y un Hamming Loss de 0.0372. La exactitud promedio y la cobertura de etiquetas
                muestran que el sistema es confiable para la clasificación automática de artículos biomédicos.
              </p>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-green-50 border border-green-200">
                  <h4 className="font-semibold text-green-800 mb-2">Fortalezas del Modelo</h4>
                  <ul className="text-sm text-green-700 space-y-1">
                    <li>• Hepatorenal: Precisión perfecta (1.000)</li>
                    <li>• Cardiovascular: F1 más alto (0.964)</li>
                    <li>• Consistencia entre todas las clases</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                  <h4 className="font-semibold text-blue-800 mb-2">Métricas Destacadas</h4>
                  <ul className="text-sm text-blue-700 space-y-1">
                    <li>• F1 Macro: 0.9463 (excelente balance)</li>
                    <li>• Average Precision: 0.9753</li>
                    <li>• Exact Match: 86.68% de casos perfectos</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
