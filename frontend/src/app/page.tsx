import Link from "next/link";

export default function Home() {
  return (
    <main className="flex h-screen items-center justify-center">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold">Construction Archive</h1>
        <p className="text-gray-400">AI-powered blueprint management system</p>
        <Link
          href="/projects/demo"
          className="inline-block bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg"
        >
          Open Demo Project
        </Link>
      </div>
    </main>
  );
}
