#define UNROLL_FACTOR  1

__kernel void program (
	float dt, 
	float eps, 
	__local float4* pblock,
	__global float4* pos_old, 
	__global float4* vel_old,  
	int N,
	float4 well,
	__global float4* force,
	float speed_limit,
	float g
	)
{

	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};

	for(int jb=0; jb < nb; jb++) 
	{ 	/* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

		int j=0; 
        for(; (j+UNROLL_FACTOR)<nt; j++) 
		{ 
		#pragma unroll UNROLL_FACTOR
		for(int k = 0; k<UNROLL_FACTOR; j++,k++) 
			{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z  + eps*eps);
			float f = p2.w * invr*invr * g;
			a += f*d; /* Accumulate acceleration */
			}
        }
		
		for(; j<nt; j++) 
		{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps*eps);
			float f = p2.w * invr*invr* g;
			a += f*d; /* Accumulate acceleration */
		}

		barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
    }
	
	
	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = v + dt*a;
	force[gti] = a;
	
	v = vel_old[gti];
	float invr = rsqrt(v.x*v.x +v.y*v.y + v.z*v.z);
	if(1/invr >speed_limit)
	{		
		//v *= invr * speed_limit;
		v *= 0.99999f;
	}
	vel_old[gti] = v ;
}


__kernel void SBODY (float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old,  
		int N, 
		float4 mouse)
{
	//const int i = get_global_id (0);

	int gti = get_global_id(0);

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};
	
         float4 p2 = mouse;
         float4 d = p2 -  p;
         float invr = rsqrt(d.x*d.x + d.y*d.y +  d.z*d.z + eps*eps);
         float f = 10000000*invr*invr*invr;
         a += f*d; /* Accumulate acceleration */

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	v += dt*a;

	if (newpos.x > 40000)
	{
		newpos.x = 40000.0f;
		v.x = 0;
		v.y = 0;
	}
		if (newpos.x < -40000)
	{
		newpos.x = -40000;
		v.x = 0;
		v.y = 0;
	}
		if (newpos.y > 40000)
	{
		newpos.y = 40000;
		v.y = 0;
		v.x = 0;
	}
		if (newpos.y <  -40000)
	{
		newpos.y = -40000;
		v.y = 0;
		v.x = 0;
	}

	vel_old[gti] = v;
	pos_old[gti] = newpos;

}

__kernel void NBODYBH2 (float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, 
		int N)
{
	//const int i = get_global_id (0);

	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};

	for(int jb=0; jb < nb; jb++) 
	{ /* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

		int j=0; 
        for(; (j+UNROLL_FACTOR)<nt; j++) 
		{ 
		#pragma unroll UNROLL_FACTOR
		for(int k = 0; k<UNROLL_FACTOR; j++,k++) 
			{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
			 }
        }
		for(; j<nt; j++) 
		{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}

       barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
      }

	  //black hole
	float4 d =  -p;
	float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
	float f = 100000 * invr*invr*invr;
	a += f*d; 

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	float4 newvel = v + dt*a;

	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = newvel;
}

__kernel void NBODY (float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, int N)
{
	//const int i = get_global_id (0);

	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};

	for(int jb=0; jb < nb; jb++) 
	{ /* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

		int j=0; 
        for(; (j+UNROLL_FACTOR)<nt; j++) 
		{ 
		#pragma unroll UNROLL_FACTOR
		for(int k = 0; k<UNROLL_FACTOR; j++,k++) 
			{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z  + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
			 }
        }
		for(; j<nt; j++) 
		{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}

       barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
      }

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = v + dt*a;
}

__kernel void NBODY2 (float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, int N)
{
	//const int i = get_global_id (0);

	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};


	for(int jb=0; jb < nb; jb++) 
	{ /* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

		int j=0; 
        for(; (j+UNROLL_FACTOR)<nt; j++) 
		{ 
		#pragma unroll UNROLL_FACTOR
		for(int k = 0; k<UNROLL_FACTOR; j++,k++) 
			{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
			 }
        }
		for(; j<nt; j++) 
		{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}

       barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
      }

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = v + dt*a;
}

__kernel void NLUF(float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, int N)
{
	const int i = get_global_id (0);

	float4 p = pos_old[i];
	float4 v = vel_old[i];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};
	
	int j = 0;
	for (; (j + UNROLL_FACTOR) < N; )
	{
	#pragma unroll UNROLL_FACTOR
		for (int k = 0; k < UNROLL_FACTOR; k++, j++)
		{
		float4 p2 = pos_old[j];
		float4 d = p2 -  p;
        float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
        float f = p2.w * invr*invr*invr*invr;
        a += f*d; /* Accumulate acceleration */
		}
	}
	for (; j < N; j++)
	{
	float4 p2 = pos_old[j];
	float4 d = p2 -  p;
    float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
    float f = p2.w * invr*invr*invr*invr;
    a += f*d; /* Accumulate acceleration */
	}

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	pos_old[i] = newpos;
	vel_old[i] = v + dt*a;
}

__kernel void _NBODY2 (float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, int N)
{
	//const int i = get_global_id (0);

	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};


	for(int jb=0; jb < nb; jb++) 
	{ /* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */
		
		//#pragma unroll
        for(int j=0; j<nt; j++) 
		{ /* For ALL cached particle positions ... */
         float4 p2 = pblock[j]; /* Read a cached particle position */
         float4 d = p2 -  p;
         float invr = rsqrt(d.x*d.x + d.y*d.y + eps);
         float f = p2.w * invr*invr*invr*invr;
         a += f*d; /* Accumulate acceleration */
        }

       barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
      }

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = v + dt*a;
}

