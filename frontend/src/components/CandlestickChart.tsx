"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, CandlestickData, Time } from "lightweight-charts";
import type { OHLCVBar, SRLevel } from "@/lib/api";

interface CandlestickChartProps {
  data: OHLCVBar[];
  supportLevels: SRLevel[];
  resistanceLevels: SRLevel[];
  height?: number;
}

export default function CandlestickChart({
  data,
  supportLevels,
  resistanceLevels,
  height = 400,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9CA3AF",
      },
      grid: {
        vertLines: { color: "#1F2937" },
        horzLines: { color: "#1F2937" },
      },
      width: containerRef.current.clientWidth,
      height,
      crosshair: {
        mode: 0, // Normal
      },
      handleScroll: {
        mouseWheel: false, // Don't hijack scroll wheel (let page scroll)
      },
      handleScale: {
        mouseWheel: false, // Don't zoom on scroll wheel
      },
      timeScale: {
        borderColor: "#374151",
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: "#374151",
      },
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22C55E",
      downColor: "#EF4444",
      borderDownColor: "#EF4444",
      borderUpColor: "#22C55E",
      wickDownColor: "#EF4444",
      wickUpColor: "#22C55E",
    });

    // Convert data to lightweight-charts format
    const chartData: CandlestickData<Time>[] = data.map((bar) => ({
      time: bar.date.split("T")[0] as string as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    candleSeries.setData(chartData);

    // Volume series
    const volumeSeries = chart.addHistogramSeries({
      color: "#4B5563",
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    const volumeData = data.map((bar) => ({
      time: bar.date.split("T")[0] as string as Time,
      value: bar.volume,
      color: bar.close >= bar.open ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)",
    }));

    volumeSeries.setData(volumeData);

    // Add S/R level lines
    for (const level of supportLevels) {
      candleSeries.createPriceLine({
        price: level.price,
        color: "#22C55E",
        lineWidth: 1,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: level.label || `S ${level.price.toFixed(2)}`,
      });
    }

    for (const level of resistanceLevels) {
      candleSeries.createPriceLine({
        price: level.price,
        color: "#EF4444",
        lineWidth: 1,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: level.label || `R ${level.price.toFixed(2)}`,
      });
    }

    // Fit content
    chart.timeScale().fitContent();

    // Resize handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, supportLevels, resistanceLevels, height]);

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg overflow-hidden"
    />
  );
}
