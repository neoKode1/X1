<script lang="ts">
	import { onMount } from 'svelte';
	import { initWs } from '$lib/ws.svelte.js';

	let { children } = $props();

	onMount(() => { initWs(); });
</script>

<svelte:head>
	<title>X1 — ARIA Dashboard</title>
	<meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

{@render children()}
