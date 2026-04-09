import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Construction Archive",
  description: "AI-powered construction blueprint viewer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-white antialiased">{children}</body>
    </html>
  );
}
